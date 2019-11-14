from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import torch.nn
import math
import cv2
import time
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import *
from pytracking.features.preprocessing import sample_patch
from pytracking.features import augmentation


class ETCOM(BaseTracker):

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize()
        self.features_initialized = True


    def initialize(self, image, state, *args, **kwargs):

        # Initialize some stuff
        self.frame_num = 1
        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize features
        self.initialize_features()

        # Check if image is color
        self.params.features.set_is_color(image.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features.get_fparams('feature_params')

        self.time = 0
        tic = time.time()

        # Get position and size
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.params.image_sample_size

        # Check if IoUNet is used
        self.use_iou_net = getattr(self.params, 'use_iou_net', True)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Set sizes
        self.img_sample_sz = torch.Tensor([self.params.image_sample_size, self.params.image_sample_size])
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        if getattr(self.params, 'score_upsample_factor', None) is None:
            self.output_sz = self.feature_sz[0]
        else:
            self.output_sz = self.params.score_upsample_factor * self.img_support_sz  # Interpolated size of the output
        self.kernel_size = self.fparams.attribute('kernel_size')

        self.iou_img_sample_sz = self.img_sample_sz

        self.params.score_fusion_strategy = getattr(self.params, 'score_fusion_strategy', 'default')
        self.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), self.output_sz.long()*self.params.effective_search_area / self.params.search_area_scale, centered=False).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)

            self.output_window = self.output_window.squeeze(0)
        # Convert image
        im = numpy_to_torch(image)
        self.im = im

        # Setup bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        x = self.generate_init_samples(im)

        self.init_classifier(x)

        if self.use_iou_net:
            self.init_iou_net()

        # Init memory
        # self.init_memory(x)

        self.time += time.time() - tic

    def track(self, image):

        self.frame_num += 1

        # Convert image
        im = numpy_to_torch(image)
        self.im = im

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.params.scale_factors
        test_x = self.extract_sample(im, self.pos, sample_scales, self.img_sample_sz)

        # Compute scores
        scores_raw = self.apply_filter(test_x)
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw)

        # Update position and scale
        if flag != 'not_found':
            if self.use_iou_net:
                update_scale_flag = getattr(self.params, 'update_scale_when_uncertain', True) or flag != 'uncertain'
                if getattr(self.params, 'use_classifier', True):
                    self.update_state(sample_pos + translation_vec)
                self.refine_target_box(sample_pos, sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif getattr(self.params, 'use_classifier', True):
                self.update_state(sample_pos + translation_vec, sample_scales[scale_ind])

        if self.params.debug >= 2:
            show_tensor(s[scale_ind,...], 5, title='Max score = {:.2f}'.format(torch.max(s[scale_ind,...]).item()))


        # ------- UPDATE ------- #

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = getattr(self.params, 'hard_negative_learning_rate', None) if hard_negative else None

        if getattr(self.params, 'update_classifier', False) and update_flag:
            # Get train sample
            train_x = TensorList([x[scale_ind:scale_ind+1, ...] for x in test_x])

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scales[scale_ind])
            train_y = self.get_label_function(sample_pos, sample_scales[scale_ind]).to(self.params.device)

            # Update the classifier model
            self.update_classifier(train_x, train_y, target_box, learning_rate, s[scale_ind,...])

            # Update memory
            # self.update_memory(train_x, train_y, learning_rate)

        # Set the pos of the tracker to iounet pos
        if self.use_iou_net and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        # Return new state
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        return new_state.tolist()


    def apply_filter(self, sample_x: TensorList):
        with torch.no_grad():
            scores = self.target_classifier.classify(self.target_filter, sample_x[0])
        return scores[...,:sample_x[0].shape[-2],:sample_x[0].shape[-1]]

    def localize_target(self, scores_raw):
        if self.params.score_fusion_strategy == 'weightedsum':
            weight = self.fparams.attribute('translation_weight')
            scores_raw = weight * scores_raw
            sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) * scores_raw.size(3))
            for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
                sf_weighted[i] = fourier.shift_fs(sf_weighted[i], math.pi * (1 - torch.Tensor([ksz[0]%2, ksz[1]%2]) / sz))

            scores_fs = fourier.sum_fs(sf_weighted)
            scores = fourier.sample_fs(scores_fs, self.output_sz)
        elif self.params.score_fusion_strategy == 'default':
            if len(scores_raw) > 1:
                raise NotImplementedError('Not implemented')
            scores = scores_raw[0]
            ksz = self.kernel_size[0]
            offset = torch.Tensor([ksz[0]%2, ksz[1]%2]) / 2
        else:
            raise ValueError('Unknown score fusion strategy.')

        if self.output_window is not None and not getattr(self.params, 'perform_hn_without_windowing', False):
            raise NotImplementedError
            scores *= self.output_window

        if getattr(self.params, 'advanced_localization', False):
            return self.localize_advanced(scores)

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        if self.params.score_fusion_strategy == 'default':
            disp = max_disp + offset
        else:
            disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
        translation_vec *= self.params.scale_factors[scale_ind]

        # Shift the score output for visualization purposes
        if self.params.debug >= 2:
            sz = scores.shape[-2:]
            scores = torch.cat([scores[...,sz[0]//2:,:], scores[...,:sz[0]//2,:]], -2)
            scores = torch.cat([scores[...,:,sz[1]//2:], scores[...,:,:sz[1]//2]], -1)

        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores):
        sz = scores.shape[-2:]

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            # raise NotImplementedError
            scores_orig = scores.clone()
            # scores_orig = torch.cat([scores_orig[..., (sz[0] + 1) // 2:, :], scores_orig[..., :(sz[0] + 1) // 2, :]], -2)
            # scores_orig = torch.cat([scores_orig[..., :, (sz[1] + 1) // 2:], scores_orig[..., :, :(sz[1] + 1) // 2]], -1)
            scores *= self.output_window

        if self.params.score_fusion_strategy == 'weightedsum':
            scores = torch.cat([scores[...,(sz[0]+1)//2:,:], scores[...,:(sz[0]+1)//2,:]], -2)
            scores = torch.cat([scores[...,:,(sz[1]+1)//2:], scores[...,:,:(sz[1]+1)//2]], -1)
            offset = torch.zeros(2)
        else:
            ksz = self.kernel_size[0]
            offset = torch.Tensor([ksz[0]%2, ksz[1]%2]) / 2

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - self.output_sz // 2
        translation_vec1 = target_disp1 * (self.img_support_sz / self.output_sz) * self.target_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'not_found'

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores = scores_orig

        # Mask out target neighborhood
        if getattr(self.params, 'use_hn_fix', False):
            target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / self.target_scale) * (self.output_sz / self.img_support_sz)
        else:
            target_neigh_sz = self.params.target_neighborhood_scale * self.target_sz / self.target_scale

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores[scale_ind:scale_ind+1,...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - self.output_sz // 2
        translation_vec2 = target_disp2 * (self.img_support_sz / self.output_sz) * self.target_scale

        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1**2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'hard_negative'

        return translation_vec1, scale_ind, scores, None


    def extract_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        return self.params.features.extract(im, pos, scales, sz)

    def get_iou_features(self):
        feat = self.params.features.get_unique_attribute('iounet_features')
        return feat

    def get_iou_backbone_features(self):
        feat = self.params.features.get_unique_attribute('iounet_backbone_features')
        return feat


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        # Compute augmentation size
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor).long().tolist()


        self.transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.params.augmentation:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation['shift']])
        if 'relativeshift' in self.params.augmentation:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.params.augmentation['relativeshift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.params.augmentation:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.params.augmentation['blur']])
        if 'scale' in self.params.augmentation:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.params.augmentation['scale']])
        if 'rotate' in self.params.augmentation:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.params.augmentation['rotate']])

        init_samples = self.params.features.extract_transformed(im, self.pos, self.target_scale, aug_expansion_sz, self.transforms)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        return init_samples

    def init_target_boxes(self):
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.pos.round(), self.target_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_label_function(self, train_x):
        # Allocate label function
        if getattr(self.params, 'update_classifier', False):
            self.y = TensorList([x.new_zeros(self.params.sample_memory_size, 1, x.shape[2]+(ksz[0]+1)%2, x.shape[3]+(ksz[1]+1)%2)
                                 for x, ksz in zip(train_x, self.kernel_size)])
        else:
            self.y = TensorList([x.new_zeros(x.shape[0], 1, x.shape[2]+(ksz[0]+1)%2, x.shape[3]+(ksz[1]+1)%2)
                                 for x, ksz in zip(train_x, self.kernel_size)])

        # Output sigma factor
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized coords
        target_center_norm = (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)

        # Generate label functions
        for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz, self.kernel_size, train_x):
            ksz_even = torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            center_pos = sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * sz
                y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center, end_pad=ksz_even)

        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0], ...] for y, x in zip(self.y, train_x)])

    def init_memory(self, train_x):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        self.init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        self.init_training_samples = train_x

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, self.fparams, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (self.pos - sample_pos) / (sample_scale * self.img_support_sz)
        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            ksz_even = torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            center = sz * target_center_norm + 0.5 * ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))
        return train_y

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)


    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates"""
        box_center = (pos - sample_pos) / sample_scale + (self.iou_img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


    def init_iou_net(self):
        # Setup IoU net and objective
        self.iou_predictor = self.params.features.get_unique_attribute('iou_predictor')
        for p in self.iou_predictor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.pos.round(), self.target_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box.clone())
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_res_features = self.get_iou_backbone_features()

        # Remove other augmentations such as rotation
        iou_res_features = TensorList([x[:target_boxes.shape[0],...] for x in iou_res_features])

        # Extract target feat
        with torch.no_grad():
            target_feat = self.iou_predictor.get_filter(iou_res_features, target_boxes)
        self.iou_filter = TensorList([x.detach().mean(0) for x in target_feat])

        if getattr(self.params, 'iounet_not_use_reference', False):
            self.iou_filter = TensorList([torch.full_like(tf, tf.norm() / tf.numel()) for tf in self.iou_filter])


    def init_classifier(self, x):
        # Get target classifier network
        self.target_classifier = self.params.features.get_unique_attribute('target_classifier')

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        self.classifier_net_name = type(self.target_classifier).__name__

        if 'Simple' in self.classifier_net_name:
            with torch.no_grad():
                target_filter = self.target_classifier.get_modulation(x[0], target_boxes)
            self.target_filter = target_filter.detach().mean(dim=0, keepdim=True)

        elif self.classifier_net_name == 'LinearFilter':
            # Generate label function
            y = self.init_label_function(x)

            if hasattr(self.params, 'num_neg_clusters'):
                self.target_classifier.num_neg_clusters = self.params.num_neg_clusters

            if hasattr(self.params, 'num_cluster_iter'):
                self.target_classifier.cluster_iter = self.params.num_cluster_iter

            if hasattr(self.params, 'use_init_filter'):
                self.target_classifier.filter_optimizer.use_init_filter = self.params.use_init_filter

            # Set number of iterations
            plot_loss = self.params.debug >= 3
            num_iter = getattr(self.params, 'net_opt_iter', None)
            # if getattr(self.params, 'update_strategy', 'default') == 'convex':
            #     plot_loss = False
            #     num_iter = 0

            with torch.no_grad():
                self.target_filter, losses = self.target_classifier.get_filter(x[0], target_boxes, y[0], num_iter=num_iter,
                                                                                   compute_losses=plot_loss, train_bb=target_boxes)

            # TODO do this better
            self.target_classifier.filter_optimizer.use_init_filter = True

            # Init memory
            if getattr(self.params, 'update_classifier', False):
                self.init_memory(x)

            if plot_loss:
                self.losses = torch.stack(losses['train'])
                plot_graph(self.losses, 10, title='Training loss')

        else:
            raise RuntimeError('Unknown target classifier type "{}"'.format(self.classifier_net_name))


    def update_classifier(self, train_x, train_y, target_box, learning_rate=None, scores=None):
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.fparams[0].learning_rate
        init_samples_weight = getattr(self.fparams[0], 'init_samples_minimum_weight', 0.0)

        if 'Simple' in self.classifier_net_name:
            raise NotImplementedError('Model update not implemented')

        elif self.classifier_net_name == 'LinearFilter' and getattr(self.params, 'update_strategy', 'default') == 'default':
            self.update_memory(train_x, train_y, target_box, learning_rate)

            low_score_th = getattr(self.params, 'low_score_opt_threshold', None)

            num_iter = 0
            if hard_negative_flag:
                num_iter = getattr(self.params, 'net_opt_hn_iter', None)
            elif low_score_th is not None and low_score_th > scores.max().item():
                num_iter = getattr(self.params, 'net_opt_low_iter', None)
            elif (self.frame_num - 1) % self.params.train_skipping == 0:
                num_iter = getattr(self.params, 'net_opt_update_iter', None)

            plot_loss = self.params.debug >= 3

            if num_iter != 0:
                samples = self.training_samples[0][:self.num_stored_samples[0],...]
                labels = self.y[0][:self.num_stored_samples[0],...]
                target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
                sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]
                with torch.no_grad():
                    self.target_filter, losses = self.target_classifier.filter_optimizer(self.target_filter, samples, labels,
                                                                                         num_iter=num_iter, compute_losses=plot_loss,
                                                                                         train_bb=target_boxes,
                                                                                         sample_weight=sample_weights)

                if plot_loss:
                    self.losses = torch.cat((self.losses, torch.stack(losses['train'])))
                    plot_graph(self.losses, 10, title='Training loss')

        elif self.classifier_net_name == 'LinearFilter' and getattr(self.params, 'update_strategy', 'default') == 'convex':

            if not hasattr(self, 'init_target_filter'):
                self.init_target_filter = self.target_filter

            num_iter = getattr(self.params, 'net_opt_update_iter', None)
            target_box = target_box.to(train_x[0].device)
            with torch.no_grad():
                target_filter_update, losses = self.target_classifier.get_modulation(train_x[0], target_box, train_y[0],
                                                                                     num_iter=num_iter,
                                                                                     compute_losses=False, train_bb=target_box)
            self.target_filter = init_samples_weight * self.init_target_filter + (1 - init_samples_weight) * \
                                    ((1 - learning_rate) * self.target_filter + learning_rate * target_filter_update)

        else:
            raise RuntimeError('Unknown target classifier type "{}"'.format(self.classifier_net_name))


    def refine_target_box(self, sample_pos, sample_scale, scale_ind, update_scale = True):
        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features()
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

            # Generate smaller boxes
            if hasattr(self.params, 'iounet_use_small_proposals') and self.params.iounet_use_small_proposals:
                if hasattr(self.params, 'iounet_use_iterative_shrinking') and self.params.iounet_use_iterative_shrinking:
                    init_center = init_box[:2] + init_box[2:] / 2
                    prev_box = init_box.clone()
                    for _ in range(self.params.iounet_num_small_proposals):
                        new_sz = prev_box[2:].clone()
                        if new_sz[1] > new_sz[0]:
                            new_sz[1] *= self.params.iounet_small_box_factor
                        else:
                            new_sz[0] *= self.params.iounet_small_box_factor

                        new_box = torch.cat((init_center - new_sz / 2, new_sz), 0).view(1,4)
                        init_boxes = torch.cat([new_box, init_boxes])
                        prev_box = new_box.squeeze()
                else:
                    if init_box[2] > init_box[3]:
                        new_width = torch.linspace(init_box[2]*self.params.iounet_small_box_factor, init_box[2],
                                                   self.params.iounet_num_small_proposals).view(-1, 1)
                        new_height = init_box[3].view(1, 1).expand(new_width.numel(), -1)
                    else:
                        new_height = torch.linspace(init_box[3] * self.params.iounet_small_box_factor, init_box[3],
                                                    self.params.iounet_num_small_proposals).view(-1, 1)
                        new_width = init_box[2].view(1, 1).expand(new_height.numel(), -1)

                    new_sz = torch.cat((new_width, new_height), 1)
                    init_center = init_box[:2] + init_box[2:]/2
                    small_proposals = torch.cat((init_center.expand(new_sz.shape[0], -1) - new_sz/2, new_sz), 1)

                    init_boxes = torch.cat([small_proposals, init_boxes])

        if not getattr(self.params, 'iounet_update_aspect_ratio', True):
            init_boxes[...,2:] = init_boxes[...,2:].prod(dim=1, keepdim=True)/init_box[2:].prod() * init_box[2:]

        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)
        # print(prof.key_averages())
        # print(prof.total_average().cuda_time_total_str)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # predict box
        k = getattr(self.params, 'iounet_k', 5)

        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        new_pos = predicted_box[:2] + predicted_box[2:]/2 - (self.iou_img_sample_sz - 1) / 2
        new_pos = new_pos.flip((0,)) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if getattr(self.params, 'use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            if hasattr(self.params, 'target_scale_update_rate'):
                self.target_scale = new_scale*self.params.target_scale_update_rate + \
                                    self.target_scale*(1 - self.params.target_scale_update_rate)
            else:
                self.target_scale = new_scale


    def optimize_boxes(self, iou_features, init_boxes):
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.iou_predictor.predict_iou(self.iou_filter, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            if getattr(self.params, 'iounet_use_log_scale', False):
                if getattr(self.params, 'iounet_update_aspect_ratio', True):
                    output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
                else:
                    scale_grad = torch.sum(bb_init.grad[:,:,2:] * bb_init[:, :, 2:], dim=2, keepdim=True)
                    scale_grad /= torch.sqrt(torch.prod(bb_init[:, :, 2:], dim=2, keepdim=True))
                    total_grad = torch.cat([bb_init.grad[:,:,:2], scale_grad, scale_grad], dim=2)
                    output_boxes = bb_init + step_length * total_grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            else:
                output_boxes = bb_init + step_length * bb_init.grad
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs[0].detach().view(-1).cpu()

    def track_sequence(self, sequence):
        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        self.initialize(image, sequence.init_state)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        frame_number = 1
        for frame in sequence.frames[1:]:
            # Set gt pos
            if frame_number < sequence.ground_truth_rect.shape[0]:
                gt_state = sequence.ground_truth_rect[frame_number, :]
                self.gt_state = gt_state
            if hasattr(self.params, 'use_gt_translation') and self.params.use_gt_translation:
                self.pos = torch.Tensor([gt_state[1] + (gt_state[3] - 1)/2, gt_state[0] + (gt_state[2] - 1)/2])

            image = self._read_image(frame)
            start_time = time.time()
            state = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)

            frame_number += 1

            if self.params.visualization:
                self.visualize(image, state)

        # print('FPS is:  {}'.format(len(sequence.frames) / self.time))
        return tracked_bb, times
