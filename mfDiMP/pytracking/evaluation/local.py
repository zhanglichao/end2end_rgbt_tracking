from pytracking.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    settings.otb_path = '/data/tracking_datasets/otb_sequences'
    settings.nfs_path = '/data/tracking_datasets/nfs'
    settings.uav_path = '/data/tracking_datasets/ECCV16_Dataset_UAV123'
    settings.tpl_path = '/data/tracking_datasets/tpc128'
    settings.vot_path = ''
    settings.trackingnet_path = ''
    settings.results_path = '/home/lichao/projects/pytracking_lichao/pytracking/tracking_results'
    settings.network_path = '/home/lichao/projects/pytracking_lichao/train_ws/checkpoints/ltr/seq_tracking'

    return settings

