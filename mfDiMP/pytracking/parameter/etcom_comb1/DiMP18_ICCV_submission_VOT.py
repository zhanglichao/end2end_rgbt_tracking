from pytracking.utils import TrackerParams, FeatureParams, Choice
from pytracking.features.extractor import MultiResolutionExtractor
from pytracking.features import trackernet
import torch

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    deep_params = TrackerParams()

    params.image_sample_size = 14*16
    params.search_area_scale = 4
    params.feature_size_odd = False

    # Learning parameters
    params.sample_memory_size = 250
    deep_params.learning_rate = 0.0075
    deep_params.init_samples_minimum_weight = 0.0
    params.train_skipping = 10
    deep_params.output_sigma_factor = 1/4

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 25
    params.net_opt_update_iter = 3
    params.net_opt_hn_iter = 3

    params.scale_factors = torch.ones(1)

    # Spatial filter parameters
    deep_params.kernel_size = (4,4)

    params.window_output = True

    # Detection parameters
    # params.score_upsample_factor = 1
    # params.score_fusion_strategy = 'weightedsum'
    # deep_params.translation_weight = 1

    # Init augmentation parameters
    # params.augmentation = {}
    params.augmentation = {'fliplr': True,
                           'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60],
                           'blur': [(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
                           'dropout': (7, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    deep_params.use_augmentation = True

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.0
    params.distractor_threshold = 100
    params.hard_negative_threshold = 0.3
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.7
    params.perform_hn_without_windowing = True

    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.iounet_augmentation = False
    params.iounet_use_log_scale = True
    params.iounet_k = 3
    params.num_init_random_boxes = 9
    params.box_jitter_pos = 0.1
    params.box_jitter_sz = 0.5
    params.maximal_aspect_ratio = 6
    params.box_refinement_iter = 5
    params.box_refinement_step_length = 1
    params.box_refinement_step_decay = 1

    deep_fparams = FeatureParams(feature_params=[deep_params])
    deep_feat = trackernet.SimpleTrackerResNet18(net_path='sdlearn_300_onlytestloss_lr_causal_mg30_iou_coco',
                                                 fparams=deep_fparams)

    params.features = MultiResolutionExtractor([deep_feat])

    params.vot_anno_conversion_type = 'preserve_area'
    return params
