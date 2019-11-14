import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor
from ltr.admin import loading
import pdb

class OptimTracker(nn.Module):
    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer, train_feature_extractor=True):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set([self.classification_layer] + self.bb_regressor_layer)))

        if not train_feature_extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, train_label, is_distractor=None, test_label=None, test_anno=None):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        num_sequences = train_imgs.shape[1]
        num_train_images = int(train_imgs.shape[0] / 2)
        num_test_images = int(test_imgs.shape[0] / 2)

        
        
        # Extract backbone features for RGB and TIR
        train_feat = self.extract_backbone_features(train_imgs[:3,...].view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs[:3,...].view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))
        train_feat_i = self.extract_backbone_features(train_imgs[3:,...].view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat_i = self.extract_backbone_features(test_imgs[3:,...].view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))


        # Concat RGB and IR feats
        # Classification features
        train_feat_clf_v = train_feat[self.classification_layer]; train_feat_clf_i = train_feat_i[self.classification_layer]
        test_feat_clf_v = test_feat[self.classification_layer]; test_feat_clf_i = test_feat_i[self.classification_layer]
        train_feat_clf = torch.cat((train_feat_clf_v, train_feat_clf_i), 1)
        test_feat_clf = torch.cat((test_feat_clf_v, test_feat_clf_i), 1)
       #train_feat_clf = torch.cat((train_feat[self.classification_layer],train_feat_i[self.classification_layer]), 1)
       #test_feat_clf = torch.cat((test_feat[self.classification_layer], test_feat_i[self.classification_layer]), 1)

        train_feat_clf = train_feat_clf.view(num_train_images, num_sequences, train_feat_clf.shape[-3], train_feat_clf.shape[-2], train_feat_clf.shape[-1])
        test_feat_clf = test_feat_clf.view(num_test_images, num_sequences, test_feat_clf.shape[-3], test_feat_clf.shape[-2], test_feat_clf.shape[-1])

        #target_scores, clf_losses = self.classifier(train_feat_clf, test_feat_clf, train_bb, train_label, is_distractor,
        #                                            test_label=test_label, test_anno=test_anno)
        target_scores, clf_losses = self.classifier(train_feat_clf, test_feat_clf, train_bb[:3], train_label[:3], is_distractor[:3],
                                                    test_label=test_label[:3], test_anno=test_anno[:3])
        
        # For clarity, send the features to bb_regressor in sequence form
        train_feat_iou = [train_feat[l].view(num_train_images, num_sequences, train_feat[l].shape[-3], train_feat[l].shape[-2],
                                   train_feat[l].shape[-1]) for l in self.bb_regressor_layer]
        test_feat_iou = [test_feat[l].view(num_test_images, num_sequences, test_feat[l].shape[-3], test_feat[l].shape[-2],
                                   test_feat[l].shape[-1]) for l in self.bb_regressor_layer]
        # train_bb = train_bb.view(num_sequences, num_train_images, 4)

        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)
        
        return target_scores, iou_pred, clf_losses

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + [self.classification_layer] if l != 'classification'])))

        # RGB: im[:3], TIR: im[3:]
        all_feat1 = self.feature_extractor(im[:,:3,...], backbone_layers)
        all_feat2 = self.feature_extractor(im[:,3:,...], backbone_layers)
        
        feat_cls = torch.cat((all_feat1[self.classification_layer], all_feat2[self.classification_layer]), 1)
        all_feat2['classification'] = self.classifier.extract_classification_feat(feat_cls)
     
        #return OrderedDict({{l: feat_cls} if l is 'classification' else {l: torch.cat((all_feat1[l], all_feat2[l]), 1)} for l in layers})
        #all_feat['classification'] = self.classifier.extract_classification_feat(all_feat[self.classification_layer])
        return OrderedDict({l: all_feat2[l] for l in layers})



@model_constructor
def steepest_descent_learn_filter_resnet18_newiou(filter_size=1, optim_iter=3, optim_init_step=1.0, optim_init_reg=0.01, output_activation=None,
                                 classification_layer='layer3', backbone_pretrained=False, clf_feat_blocks=1,
                                 clf_feat_norm=True, init_filter_norm=False, final_conv=False,
                                 out_feature_dim=256, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, test_loss=None,
                                           mask_init_factor=4.0, iou_input_dim=(256,256), iou_inter_dim=(256,256),
                                                  jitter_sigma_factor=None, train_backbone=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm, feature_dim=out_feature_dim)
    optimizer = clf_optimizer.SteepestDescentLearn(num_iter=optim_iter, filter_size=filter_size, init_step_length=optim_init_step,
                                                   init_filter_reg=optim_init_reg, feature_dim=out_feature_dim,
                                                   init_gauss_sigma=init_gauss_sigma, num_dist_bins=num_dist_bins,
                                                   bin_displacement=bin_displacement, test_loss=test_loss, mask_init_factor=mask_init_factor)
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor,
                                         output_activation=output_activation, jitter_sigma_factor=jitter_sigma_factor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = OptimTracker(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                       classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'], train_feature_extractor=train_backbone)
    return net



@model_constructor
def steepest_descent_learn_filter_resnet50_newiou(filter_size=1, optim_iter=3, optim_init_step=1.0, optim_init_reg=0.01, output_activation=None,
                                 classification_layer='layer3', backbone_pretrained=False, clf_feat_blocks=1,
                                 clf_feat_norm=True, init_filter_norm=False, final_conv=False,
                                 out_feature_dim=256, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, test_loss=None,
                                           mask_init_factor=4.0, iou_input_dim=(256,256), iou_inter_dim=(256,256),
                                                  jitter_sigma_factor=None):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    
    # classifier
    clf_feature_extractor = clf_features.residual_bottleneck_comb(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm, feature_dim=out_feature_dim)
    optimizer = clf_optimizer.SteepestDescentLearn(num_iter=optim_iter, filter_size=filter_size, init_step_length=optim_init_step,
                                                   init_filter_reg=optim_init_reg, feature_dim=out_feature_dim,
                                                   init_gauss_sigma=init_gauss_sigma, num_dist_bins=num_dist_bins,
                                                   bin_displacement=bin_displacement, test_loss=test_loss, mask_init_factor=mask_init_factor)
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor,
                                         output_activation=output_activation, jitter_sigma_factor=jitter_sigma_factor)    
    # Bounding box regressor
    # combine RGB and TIR by 2*
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)
    # load pretrained model
    pretrainmodel_path='/home/lichao/projects/pytracking_lichao/pytracking/DiMP_nets/sdlearn_300_onlytestloss_lr_causal_mg30_iou_nocf_res50_lfilt512_coco/OptimTracker_ep0040.pth.tar'
    pretrainmodel = loading.torch_load_legacy(pretrainmodel_path)['net']
    usepretrain = True; updback = True; updcls = True; updbb = True

    if usepretrain:
        if updback:
            # update backbone
            backbone_dict = backbone_net.state_dict()
            pretrain_dict = {k[len('feature_extractor.'):]: v for k, v in pretrainmodel.items() if k[len('feature_extractor.'):] in backbone_dict}
            backbone_net.load_state_dict(pretrain_dict)

        if updcls:
            # update classifier
            pretrainmodel['classifier.feature_extractor.0.weight']=torch.cat((pretrainmodel['classifier.feature_extractor.0.weight'],pretrainmodel['classifier.feature_extractor.0.weight']),1)
            classifier_dict = classifier.state_dict()
            pretrain_dict = {k[len('classifier.'):]: v for k, v in pretrainmodel.items() if k[len('classifier.'):] in classifier_dict}
            #classifier_dict.update(pretrain_dict)
            classifier.load_state_dict(pretrain_dict)
        if updbb:
            # update Bounding box regressor
            
            bb_regressor_dict = bb_regressor.state_dict()
            pretrain_dict = {k[len('bb_regressor.'):]: v for k, v in pretrainmodel.items() if k[len('bb_regressor.'):] in bb_regressor_dict}
            bb_regressor.load_state_dict(pretrain_dict)

    net = OptimTracker(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                       classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net
