class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/lichao/projects/pytracking_lichao/train_ws/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/media/martin/Samsung_T5/tracking_datasets/LaSOTBenchmark/'
        self.got10k_dir = '/home/lichao/tracking/datasets/GOT-10k/train/'
        self.got10k_i_dir = '/home/lichao/tracking/datasets/GOT-10k_i_/train/'
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
