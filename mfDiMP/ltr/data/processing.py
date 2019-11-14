import torch
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils
import ltr.data.bounding_box_utils as bbutils
import numpy as np


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test':  transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class ATOMProcessing(BaseProcessing):
    """ The processing class used for training ATOM. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                             sigma_factor=self.proposal_params['sigma_factor']
                                                             )

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals'-
                'proposal_iou'  -
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'] = [self.transform[s](x) for x in crops]
            data[s + '_anno'] = boxes

        # Generate proposals
        frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = list(frame2_proposals)
        data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data



class TrackingProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, mode='pair',
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        # Mode is either sequence or pair
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])

        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, frame2_gt_crop):
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        frame2_proposals = np.zeros((num_proposals, 4))
        gt_iou = np.zeros(num_proposals)
        sample_p = np.zeros(num_proposals)

        for i in range(num_proposals):
            frame2_proposals[i, :], gt_iou[i], sample_p[i] = bbutils.perturb_box(
                frame2_gt_crop,
                min_iou=self.proposal_params['min_iou'],
                sigma_factor=self.proposal_params['sigma_factor']
            )

        gt_iou = gt_iou * 2 - 1

        return frame2_proposals, gt_iou

    def _generate_label_function(self, target_bb, is_distractor=None):
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))
        
        if is_distractor is not None:
            gauss_label *= (1 - is_distractor).view(-1, 1, 1).float()
        return gauss_label

    def _generate_box_segmentation(self, target_bb, im_sz, is_distractor=None):
        bb = target_bb.view(-1, 4)
        seg = torch.zeros(bb.shape[0], im_sz[0], im_sz[1], dtype=torch.uint8)
        for i in range(bb.shape[0]):
            if is_distractor is None or is_distractor[i] == 0:
                seg[i, max(int(bb[i,1]),0):min(int(bb[i,1]+bb[i,3]),im_sz[0]), max(int(bb[i,0]),0):min(int(bb[i,0]+bb[i,2]),im_sz[1])] = 1

        return seg

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.jittered_center_crop_comb(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                self.search_area_factor, self.output_sz)

            data[s + '_images'] = [self.transform[s](x) for x in crops]
            data[s + '_anno'] = boxes

        if self.proposal_params:
            frame2_proposals, gt_iou = zip(*[self._generate_proposals(a.numpy()) for a in data['test_anno']])

            data['test_proposals'] = [torch.tensor(p, dtype=torch.float32) for p in frame2_proposals]
            data['proposal_iou'] = [torch.tensor(gi, dtype=torch.float32) for gi in gt_iou]

        if 'is_distractor_test_frame' in data:
            data['is_distractor_test_frame'] = torch.tensor(data['is_distractor_test_frame'], dtype=torch.uint8)
        else:
            data['is_distractor_test_frame'] = torch.zeros(len(data['test_images']), dtype=torch.uint8)

        if 'is_distractor_train_frame' in data:
            data['is_distractor_train_frame'] = torch.tensor(data['is_distractor_train_frame'], dtype=torch.uint8)
        else:
            data['is_distractor_train_frame'] = torch.zeros(len(data['train_images']), dtype=torch.uint8)

        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'], data['is_distractor_train_frame'])

            data['test_label'] = self._generate_label_function(data['test_anno'], data['is_distractor_test_frame'])

        return data
