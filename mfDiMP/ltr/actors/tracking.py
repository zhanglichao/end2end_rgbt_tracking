from . import BaseActor
import torch
import torch.nn.functional as F


class OptimTrackerActor(BaseActor):
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0, 'train_clf': 1.0, 'init_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):

        # Run network
        target_scores, iou_pred, clf_losses = self.net(data['train_images'],
                                                       data['test_images'],
                                                       data['train_anno'],
                                                       data['test_proposals'],
                                                       data['train_label'],
                                                       data['is_distractor_train_frame'],
                                                       test_label=data['test_label'],
                                                       test_anno=data['test_anno'])

        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        is_distractor_test = data['is_distractor_test_frame'].view(-1)

        iou_pred_valid = iou_pred.view(-1, iou_pred.shape[2])[is_distractor_test == 0, :]
        iou_gt_valid = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])[is_distractor_test == 0, :]

        # Compute loss
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred_valid, iou_gt_valid)
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test
        loss_init_clf = self.loss_weight['init_clf'] * clf_losses['train'][0]
        loss_train_clf = self.loss_weight['train_clf'] * clf_losses['train'][-1]

        loss_iter_clf = 0
        if 'iter_clf' in self.loss_weight.keys():
            loss_iter_clf = (self.loss_weight['iter_clf'] / (len(clf_losses['train']) - 2)) * sum(clf_losses['train'][1:-1])

        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses['test'][0]

        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            loss_test_iter_clf = (self.loss_weight['test_iter_clf'] / (len(clf_losses['test']) - 2)) * sum(clf_losses['test'][1:-1])

        loss = loss_iou + loss_target_classifier + loss_init_clf + loss_train_clf + loss_iter_clf + loss_test_init_clf + loss_test_iter_clf

        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item(),
                 'Loss/init_clf': loss_init_clf.item(),
                 'Loss/train_clf': loss_train_clf.item()}

        if 'iter_clf' in self.loss_weight.keys():
            stats['Loss/iter_clf'] = loss_iter_clf.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()

        stats['ClfTrain/init_loss'] = clf_losses['train'][0].item()
        stats['ClfTrain/train_loss'] = clf_losses['train'][-1].item()
        if len(clf_losses['train']) > 2:
            stats['ClfTrain/iter_loss'] = sum(clf_losses['train'][1:-1]).item() / (len(clf_losses['train']) - 2)

        stats['ClfTrain/test_loss'] = clf_loss_test.item()

        if len(clf_losses['test']) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses['test'][0].item()
            if len(clf_losses['test']) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses['test'][1:-1]).item() / (len(clf_losses['test']) - 2)

        return loss, stats

