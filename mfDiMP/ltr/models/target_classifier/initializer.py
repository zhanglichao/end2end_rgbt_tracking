import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.layers.blocks import conv_block
import math


class FilterPool(nn.Module):
    def __init__(self, filter_size=1, feature_stride=16, pool_square=False):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(filter_size, filter_size, 1/feature_stride)
        self.pool_square = pool_square

    def forward(self, feat, bb):
        # Add batch_index to rois
        bb = bb.view(-1,4)
        num_images_total = bb.shape[0]
        batch_index = torch.arange(num_images_total, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        pool_bb = bb.clone()

        if self.pool_square:
            bb_sz = pool_bb[:, 2:4].prod(dim=1, keepdim=True).sqrt()
            pool_bb[:, :2] += pool_bb[:, 2:]/2 - bb_sz/2
            pool_bb[:, 2:] = bb_sz

        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        roi1 = torch.cat((batch_index, pool_bb), dim=1)

        return self.prroi_pool(feat, roi1)


class FilterInitializerLinear(nn.Module):
    def __init__(self, filter_size=1, feature_dim=256, feature_stride=16, pool_square=False, filter_norm=True,
                 conv_ksz=3):
        super().__init__()
        
        self.filter_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=conv_ksz, padding=conv_ksz // 2)
        self.filter_pool = FilterPool(filter_size=filter_size, feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, feat, bb):
        """Initialize filter.
        feat: (images_in_sequence, sequences, feat_dim, H, W)
        bb: (images_in_sequence, sequences, 4)
        output: (sequences, feat_dim, fH, fW)"""

        num_images = feat.shape[0]        
        
        feat = self.filter_conv(feat.view(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1]))

        weights = self.filter_pool(feat, bb)

        if num_images > 1:
            weights = torch.mean(weights.view(num_images, -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)

        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])

        return weights


