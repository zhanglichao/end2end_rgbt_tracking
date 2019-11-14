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
        num_train_images = train_imgs.shape[0]
        num_test_images = test_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # imgs = torch.cat((train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]),
        #                   test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1])))
        # feat = self.extract_backbone_features(imgs)
        # sid = num_sequences*num_train_images
        # train_feat = OrderedDict({k: f[:sid,...] for k, f in feat.items()})
        # test_feat = OrderedDict({k: f[sid:,...] for k, f in feat.items()})

        # Classification features
        train_feat_clf = train_feat[self.classification_layer]
        test_feat_clf = test_feat[self.classification_layer]
        train_feat_clf = train_feat_clf.view(num_train_images, num_sequences, train_feat_clf.shape[-3], train_feat_clf.shape[-2], train_feat_clf.shape[-1])
        test_feat_clf = test_feat_clf.view(num_test_images, num_sequences, test_feat_clf.shape[-3], test_feat_clf.shape[-2], test_feat_clf.shape[-1])

        target_scores, clf_losses = self.classifier(train_feat_clf, test_feat_clf, train_bb, train_label, is_distractor,
                                                    test_label=test_label, test_anno=test_anno)

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
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.classifier.extract_classification_feat(all_feat[self.classification_layer])
        return OrderedDict({l: all_feat[l] for l in layers})



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
    clf_feature_extractor = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
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
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = OptimTracker(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                       classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net
