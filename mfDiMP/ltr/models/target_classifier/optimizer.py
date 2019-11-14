import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.models.layers.blocks import conv_block
import ltr.models.layers.filter as filter_layer
from ltr.models.layers.distance import DistanceMap
import math
from pytracking.libs import dcf, fourier, complex
import ltr.models.loss as ltr_losses



class SteepestDescentLearn(nn.Module):
    def __init__(self, num_iter=1, filter_size=1, feature_dim=256, feat_stride=16, init_step_length=1.0,
                 init_filter_reg=1e-2, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0,
                 test_loss=None):
        super().__init__()

        if test_loss is None:
            test_loss = ltr_losses.LBHinge(threshold=0.05)

        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.num_iter = num_iter
        self.test_loss = test_loss
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.feat_stride = feat_stride
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)

        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).view(1,-1,1,1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0,0,0,0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        self.target_mask_predictor = nn.Sequential(nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False), nn.Sigmoid())
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d)

        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)


    def forward(self, filter, feat, label, compute_losses=True, sample_weight=None, num_iter=None, train_bb=None, is_distractor=None, test_feat=None, test_label=None, test_anno=None):
        if num_iter is None:
            num_iter = self.num_iter

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (filter.shape[-2], filter.shape[-1])

        step_length = torch.exp(self.log_step_length)
        reg_weight = self.filter_reg*self.filter_reg

        # Compute distance map
        center = ((train_bb[..., :2] + train_bb[..., 2:] / 2) / self.feat_stride).view(-1, 2).flip((1,))
        if is_distractor is not None:
            center[is_distractor.view(-1), :] = 99999
        dist_map = self.distance_map(center, label.shape[-2:])

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).view(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        target_mask = self.target_mask_predictor(dist_map).view(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        spatial_weight = self.spatial_weight_predictor(dist_map).view(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])

        background_mask = 1.0 - target_mask
        if sample_weight is None:
            sample_weight = (1.0 / feat.shape[0]) * (spatial_weight * spatial_weight)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.view(-1, 1, 1, 1) * (spatial_weight * spatial_weight)

        losses = {'train': [], 'test': []}

        for i in range(num_iter):
            # Compute gradient
            scores = filter_layer.apply_filter(feat, filter)
            scores = target_mask * scores + background_mask * F.relu(scores)
            score_mask = (scores.detach() > 0).float() * background_mask + target_mask
            residuals = sample_weight * (scores - label_map)
            filter_grad = filter_layer.apply_feat_transpose(feat, residuals, filter_sz, training=self.training) + \
                          reg_weight * filter

            # Map the gradient
            scores_grad = filter_layer.apply_filter(feat, filter_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)
            filter_q = filter_layer.apply_feat_transpose(feat, scores_grad, filter_sz, training=self.training) + \
                       reg_weight * filter_grad

            # Compute step length
            alpha_num = (filter_grad * filter_grad).view(filter.shape[0], -1).sum(dim=1)
            alpha_den = (filter_grad * filter_q).view(filter.shape[0], -1).sum(dim=1).abs().clamp(1e-4)
            alpha = alpha_num / alpha_den

            # Update filter
            filter = filter - (step_length * alpha.view(-1,1,1,1)) * filter_grad

            if compute_losses:
                losses['train'].append((sample_weight * (scores - label_map)**2).mean())
                if test_feat is not None:
                    losses['test'].append(self._compute_test_loss(filter, test_feat, test_label, test_anno))

        if compute_losses:
            scores = filter_layer.apply_filter(feat, filter)
            scores = target_mask * scores + background_mask * F.relu(scores)
            losses['train'].append((sample_weight * (scores - label_map)**2).mean())
            if test_feat is not None:
                losses['test'].append(self._compute_test_loss(filter, test_feat, test_label, test_anno))

        return filter, losses

    def _compute_test_loss(self, filter, feat, label, target_bb=None):
        scores = filter_layer.apply_filter(feat, filter)
        return self.test_loss(scores, label, target_bb)


