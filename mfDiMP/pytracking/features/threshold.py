import torch
from pytracking.features.featurebase import FeatureBase, MultiFeatureBase
from pytracking import TensorList
import pdb
import numpy as np

class Motion(MultiFeatureBase):
    """Motion feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 1

    def initialize(self):
        
        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1]

    def stride(self):
        ss = getattr(self.fparams,'feature_params')[0].cell_size
        return TensorList([s * ss for s in self.pool_stride])

    def extract(self, im: torch.Tensor):
        
        thresh_im = im[:,3:6,...]/255 - im[:,6:,...]/255
        thresh_im = torch.abs(thresh_im)
        binary_im = thresh_im > fparam.threshold
        thresh_im = binary_im.float()

        thresh_feature_scale = 5.0
        thresh_im = thresh_feature_scale * thresh_im

        thresh_im = average_feature_region(thresh_im, cell_size)

        return TensorList([thresh_im])

    def extract_comb(self, im: torch.Tensor):
        im = im.cuda()
        thresh_im = im[:,3:4,...]/255 - im[:,6:7,...]/255
        thresh_im = torch.abs(thresh_im)
        threshold = getattr(self.fparams,'feature_params')[0].threshold
        binary_im = thresh_im > threshold
        thresh_im = binary_im.float()

        thresh_feature_scale = getattr(self.fparams,'feature_params')[0].thresh_feature_scale
        thresh_im = thresh_feature_scale * thresh_im

        cell_size = getattr(self.fparams,'feature_params')[0].cell_size
        thresh_im = average_feature_region(thresh_im, cell_size)
        
        return TensorList([thresh_im])


def average_feature_region(im, region_size):

    region_area = region_size**2
    maxval = 1.0
    
    iImage = integralVecImage(im)

    # region indices
    #i1 = [*range(region_size, im.size(2), region_size)]
    #i2 = [*range(region_size, im.size(3), region_size)]
    
    i1 = np.arange(region_size, iImage.size(2), region_size)
    i2 = np.arange(region_size, iImage.size(3), region_size)
    i1_ = i1-region_size; i2_ = i2-region_size

    region_image = (iImage[:,:,i1,:][...,i2] - iImage[:,:,i1,:][...,i2_] - iImage[:,:,i1_,:][...,i2] + iImage[:,:,i1_,:][...,i2_]) / (region_area * maxval)
    #region_image = (iImage[:,:,i1,i2] - iImage[:,:,i1,i2-region_size] - iImage[:,:,i1-region_size,i2] + iImage[:,:,i1-region_size,i2-region_size]) / (region_area * maxval)

    return region_image

def integralVecImage(I):     

    intImage = I.new_zeros(I.size(0), I.size(1), I.size(2)+1, I.size(3)+1) # , dtype=I.dtype
    intImage[:, :, 1:, 1:] = I.cumsum(2).cumsum(3)

    return intImage
