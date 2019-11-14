import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.models.layers.blocks import conv_block
import ltr.models.layers.filter as filter_layer
import math



class LinearFilter(nn.Module):
    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None, output_activation=None, jitter_sigma_factor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.output_activation = output_activation
        self.jitter_sigma_factor = jitter_sigma_factor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, train_label, is_distractor=None, test_label=None, test_anno=None):
        """Order of dimensions should always be images_in_sequence before sequence."""

        # Extract features
        train_feat = self.extract_classification_feat(train_feat)
        test_feat = self.extract_classification_feat(test_feat)

        # Train filter
        if self.jitter_sigma_factor is not None:
            train_bb_optim = train_bb.clone()
            train_bb_optim[..., 0:2] = train_bb_optim[..., 0:2] + torch.randn_like(train_bb_optim[..., 0:2])*\
                                       train_bb_optim[..., 2:].prod(dim=-1, keepdim=True).sqrt()*self.jitter_sigma_factor
        else:
            train_bb_optim = train_bb
        filter, losses = self.get_filter(train_feat, train_bb, train_label, is_distractor=is_distractor,
                                         train_bb=train_bb_optim, test_feat=test_feat, test_label=test_label, test_anno=test_anno)

        # Classify samples
        test_scores = self.classify(filter, test_feat)

        return test_scores, losses

    def extract_classification_feat(self, feat):
        if self.feature_extractor is None:
            return feat
        if feat.dim() == 4:
            return self.feature_extractor(feat)

        num_images = feat.shape[0]
        num_sequences = feat.shape[1]
        output = self.feature_extractor(feat.view(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1]))
        return output.view(num_images, num_sequences, output.shape[-3], output.shape[-2], output.shape[-1])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        if self.output_activation is not None:
            scores = self.output_activation(scores)

        return scores

    def get_filter(self, feat, bb, label, is_distractor=None, **kwargs):
        if is_distractor is not None:
            is_distractor = is_distractor.view(-1)
            num_sequences = feat.shape[1]

            # Note: A bit annoying since we assume here that every sequence has the same number of distractors
            feat_target = feat.view(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1])[is_distractor == 0, ...].\
                view(-1, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
            bb_target = bb.view(-1, 4)[is_distractor == 0, ...].view(-1, num_sequences, 4)

            weights = self.filter_initializer(feat_target, bb_target)
        else:
            weights = self.filter_initializer(feat, bb)

        weights, losses = self.filter_optimizer(weights, feat, label, is_distractor=is_distractor, **kwargs)
        return weights, losses

