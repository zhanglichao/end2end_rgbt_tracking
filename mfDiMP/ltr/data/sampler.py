import os
import random
import torch.utils.data
from pytracking import TensorDict
import itertools


def no_processing(data):
    return data


class ATOMSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a train frame, used to obtain the modulation vector, and ii) a set of test frames on which
    the IoU prediction loss is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A 'train frame' is then sampled randomly from the sequence. Next, depending on the
    frame_sample_mode, the required number of test frames are sampled randomly, either  from the range
    [train_frame_id - max_gap, train_frame_id + max_gap] in the 'default' mode, or from [train_frame_id, train_frame_id + max_gap]
    in the 'causal' mode. Only the frames in which the target is visible are sampled, and if enough visible frames are
    not found, the 'max_gap' is incremented.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap, num_test_frames=1, processing=no_processing,
                 frame_sample_mode='default'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train (reference) frame and the test frames.
            num_test_frames - Number of test frames used for calculating the IoU prediction loss.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'default' or 'causal'. If 'causal', then the test frames are sampled in a causal
                                manner.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = 1                         # Only a single train frame allowed
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        min_visible_frames = 2 * (self.num_test_frames + self.num_train_frames)
        enough_visible_frames = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not enough_visible_frames:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)
            anno, visible = dataset.get_sequence_info(seq_id)
            num_visible = visible.type(torch.int64).sum().item()
            enough_visible_frames = not is_video_dataset or (num_visible > min_visible_frames and len(visible) >= 20)

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0
            if self.frame_sample_mode == 'default':
                # Sample frame numbers
                while test_frame_ids is None:
                    train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames)
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    gap_increase += 5   # Increase gap until a frame is found
            elif self.frame_sample_mode == 'causal':
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible)-self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    gap_increase += 5   # Increase gap until a frame is found
            else:
                raise ValueError('Unknown frame_sample_mode.')
        else:
            train_frame_ids = [1]*self.num_train_frames
            test_frame_ids = [1]*self.num_test_frames

        # Get frames
        train_frames, train_anno, _ = dataset.get_frames(seq_id, train_frame_ids, anno)
        test_frames, test_anno, _ = dataset.get_frames(seq_id, test_frame_ids, anno)

        # Prepare data
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'dataset': dataset.get_name()})

        # Send for processing
        return self.processing(data)

class RandomSequenceWithDistractors(torch.utils.data.Dataset):
    """
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_seq_test_frames, num_class_distractor_frames=0, num_random_distractor_frames=0,
                 num_seq_train_frames=1, num_class_distractor_train_frames=0, num_random_distractor_train_frames=0,
                 processing=no_processing, parent_class_list=None, sample_mode='sequence',
                 frame_sample_mode='default', max_distractor_gap=9999999):

        self.use_class_info = num_class_distractor_train_frames > 0 or num_class_distractor_frames > 0
        if self.use_class_info:
            for d in datasets:
                assert d.has_class_info(), 'Dataset must have class info'

        assert num_class_distractor_frames >= num_class_distractor_train_frames, 'Cannot have >1 train frame per distractor'
        assert num_random_distractor_frames >= num_random_distractor_train_frames, 'Cannot have >1 train frame per distractor'

        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_seq_test_frames = num_seq_test_frames
        self.num_class_distractor_frames = num_class_distractor_frames
        self.num_random_distractor_frames = num_random_distractor_frames
        self.num_seq_train_frames = num_seq_train_frames
        self.num_class_distractor_train_frames = num_class_distractor_train_frames
        self.num_random_distractor_train_frames = num_random_distractor_train_frames
        self.processing = processing
        # self.map_parent = class_map
        self.parent_class_list = parent_class_list
        self.sample_mode = sample_mode
        self.frame_sample_mode = frame_sample_mode
        self.max_distractor_gap = max_distractor_gap

        self.all_dataset_classes = []
        # construct the class map here
        for dataset in datasets:
            if dataset.get_name() == 'lasot':
                self.all_dataset_classes = self.all_dataset_classes + dataset.class_list_proper
            else:
                self.all_dataset_classes = self.all_dataset_classes + dataset.class_list
            # self.all_dataset_classes.append(class_item for class_item in dataset.class_list)

        self.map_parent = dict()

    def __len__(self):
        return self.samples_per_epoch

    # def _get_parent_classList(self):
    #     project_path = os.path.abspath(os.path.dirname(__file__))
    #     file_path = os.path.join(project_path, '../data_specs/parent_class_imagenet_extended.txt')
    #
    #     # load the parent class file -> refer to the imagenet website for the list of parent classes
    #     f = open(file_path)
    #     major_classes = list(csv.reader(f))
    #     f.close()
    #
    #     parent_classes = [cls[0] for cls in major_classes]
    #     parent_classes.append('other')
    #     return parent_classes

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def _dict_cat(self, *dicts):
        dict_cat = {}
        for key in list(dicts[0].keys()):
            val_list = [d[key] for d in dicts if isinstance(d, dict)]
            dict_cat[key] = list(itertools.chain.from_iterable(val_list))

        return dict_cat

    def _sample_class_distractors(self, dataset, sampled_seq, sequences, num_test_frames, num_train_frames):
        cls_dist_train_frames = []
        cls_dist_train_anno = []
        cls_dist_test_frames = []
        cls_dist_test_anno = []

        i = 0
        while i < num_test_frames:
            dist_seq_id = random.choices(sequences)[0]
            while dist_seq_id == sampled_seq:
                dist_seq_id = random.choices(sequences)[0]

            dist_seq_info_dict = dataset.get_sequence_info(dist_seq_id)
            visible = dist_seq_info_dict['visible']

            dist_train_frame_id = self._sample_visible_ids(visible)
            if dist_train_frame_id is None:
                continue

            dist_test_frame_id = self._sample_visible_ids(visible, min_id=dist_train_frame_id[0] - self.max_distractor_gap,
                                                          max_id=dist_train_frame_id[0] + self.max_distractor_gap)
            if dist_test_frame_id is None:
                continue

            frame, anno_dict, _ = dataset.get_frames(dist_seq_id, dist_test_frame_id, dist_seq_info_dict)

            cls_dist_test_frames += frame
            cls_dist_test_anno = self._dict_cat(anno_dict, cls_dist_test_anno)

            if i < num_train_frames:
                frame, anno_dict, _ = dataset.get_frames(dist_seq_id, dist_train_frame_id, dist_seq_info_dict)
                cls_dist_train_frames += frame
                cls_dist_train_anno = self._dict_cat(anno_dict, cls_dist_train_anno)

            i += 1

        return cls_dist_train_frames, cls_dist_train_anno, cls_dist_test_frames, cls_dist_test_anno

    def _sample_random_distractors(self, num_test_frames, num_train_frames):
        rnd_dist_train_frames = []
        rnd_dist_train_anno = []
        rnd_dist_test_frames = []
        rnd_dist_test_anno = []

        i = 0
        while i < num_test_frames:
            dist_dataset = random.choices(self.datasets, self.p_datasets)[0]
            dist_seq_id = random.randint(0, dist_dataset.get_num_sequences() - 1)

            dist_seq_info_dict = dist_dataset.get_sequence_info(dist_seq_id)
            visible = dist_seq_info_dict['visible']

            dist_train_frame_id = self._sample_visible_ids(visible)
            dist_test_frame_id = self._sample_visible_ids(visible)

            if dist_test_frame_id is None:
                continue
            frame, anno_dict, _ = dist_dataset.get_frames(dist_seq_id, dist_test_frame_id, dist_seq_info_dict)

            rnd_dist_test_frames += frame
            rnd_dist_test_anno = self._dict_cat(rnd_dist_test_anno, anno_dict)

            if i < num_train_frames:
                frame, anno_dict, _ = dist_dataset.get_frames(dist_seq_id, dist_train_frame_id, dist_seq_info_dict)
                rnd_dist_train_frames += frame
                rnd_dist_train_anno = self._dict_cat(rnd_dist_train_anno, anno_dict)

            i += 1

        return rnd_dist_train_frames, rnd_dist_train_anno, rnd_dist_test_frames, rnd_dist_test_anno

    def __getitem__(self, index):
        """
        Args:
            index (int): Index (Ignored since we sample randomly)

        Returns:

        """

        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        enough_visible_frames = False
        # TODO clean this part
        while not enough_visible_frames:
            # Select a class
            if self.sample_mode == 'sequence':
                while not enough_visible_frames:
                    # Sample a sequence
                    seq_id = random.randint(0, dataset.get_num_sequences() - 1)
                    # Sample frames
                    seq_info_dict = dataset.get_sequence_info(seq_id)
                    visible = seq_info_dict['visible']

                    enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (self.num_seq_test_frames + self.num_seq_train_frames) and \
                        len(visible) >= 20

                    enough_visible_frames = enough_visible_frames or not is_video_dataset
                if self.use_class_info:
                    class_name = dataset.get_class_name(seq_id)
                    class_sequences = dataset.get_sequences_in_class(class_name)
            elif self.sample_mode == 'class':
                class_name = random.choices(dataset.get_class_list())[0]
                class_sequences = dataset.get_sequences_in_class(class_name)

                # Sample test frames from the sequence
                try_ct = 0
                while not enough_visible_frames and try_ct < 5:
                    # Sample a sequence
                    seq_id = random.choices(class_sequences)[0]
                    # Sample frames
                    seq_info_dict = dataset.get_sequence_info(seq_id)
                    visible = seq_info_dict['visible']

                    # TODO probably filter sequences where we don't have enough visible frames in a pre-processing step
                    #  so that we are not stuck in a while loop
                    enough_visible_frames = visible.type(torch.int64).sum().item() > self.num_seq_test_frames + \
                                            self.num_seq_train_frames
                    enough_visible_frames = enough_visible_frames or not is_video_dataset
                    try_ct += 1
            else:
                raise ValueError

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0
            if self.frame_sample_mode == 'default':
                while test_frame_ids is None:
                    train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_seq_train_frames)
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_seq_test_frames)
                    gap_increase += 5   # Increase gap until a frame is found
            elif self.frame_sample_mode == 'causal':
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_seq_train_frames-1,
                                                             max_id=len(visible)-self.num_seq_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_seq_train_frames-1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_seq_test_frames)
                    gap_increase += 5   # Increase gap until a frame is found
            else:
                raise ValueError('Unknown frame_sample_mode.')
        else:
            train_frame_ids = [1]*self.num_seq_train_frames
            test_frame_ids = [1]*self.num_seq_test_frames

        seq_train_frames, seq_train_anno, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)

        seq_test_frames, seq_test_anno, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        assert meta_obj_train['object_class'] == meta_obj_test['object_class'], "Train and test classes don't match!!"

        # Sample from sequences with the same class
        # TODO fix sequences which do not have a single visible frame
        if self.use_class_info and len(class_sequences) > 5:
            cls_dist_train_frames, cls_dist_train_anno, cls_dist_test_frames, cls_dist_test_anno = \
                self._sample_class_distractors(dataset, seq_id, class_sequences, self.num_class_distractor_frames,
                self.num_class_distractor_train_frames)
            num_rnd_distractors = self.num_random_distractor_frames
            num_rnd_train_distractors = self.num_random_distractor_train_frames
        else:
            cls_dist_train_frames = []
            cls_dist_train_anno = []
            cls_dist_test_frames = []
            cls_dist_test_anno = []
            num_rnd_distractors = self.num_random_distractor_frames + self.num_class_distractor_frames
            num_rnd_train_distractors = self.num_random_distractor_train_frames + self.num_class_distractor_train_frames

        # Sample sequences from any class
        rnd_dist_train_frames, rnd_dist_train_anno, rnd_dist_test_frames, rnd_dist_test_anno = \
            self._sample_random_distractors(num_rnd_distractors, num_rnd_train_distractors)

        train_frames = seq_train_frames + cls_dist_train_frames + rnd_dist_train_frames
        test_frames = seq_test_frames + cls_dist_test_frames + rnd_dist_test_frames

        train_anno = self._dict_cat(seq_train_anno, cls_dist_train_anno, rnd_dist_train_anno)
        test_anno = self._dict_cat(seq_test_anno, cls_dist_test_anno, rnd_dist_test_anno)

        is_distractor_train_frame = [False]*self.num_seq_train_frames + \
                                    [True]*(self.num_class_distractor_train_frames + self.num_random_distractor_train_frames)
        is_distractor_test_frame = [False]*self.num_seq_test_frames + [True]*(self.num_class_distractor_frames +
                                                                       self.num_random_distractor_frames)
        if self.parent_class_list:
            if meta_obj_train['object_class']:
                parent_class = self.map_parent[meta_obj_train['object_class']]
                parent_class_id = self.parent_class_list.index(parent_class)
            else:
                parent_class = None
                parent_class_id = -1
        else:
            parent_class = None
            parent_class_id = -1

        # TODO send in class name for each frame
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_anno': test_anno['bbox'],
                           'object_class_id': parent_class_id,
                           'motion_class': meta_obj_train['motion_class'],
                           'major_class': meta_obj_train['major_class'],
                           'root_class': meta_obj_train['root_class'],
                           'motion_adverb': meta_obj_train['motion_adverb'],
                           'object_class_name': meta_obj_train['object_class'],
                           # 'object_class_name': parent_class,
                           'dataset': dataset.get_name(),
                           'is_distractor_train_frame': is_distractor_train_frame,
                           'is_distractor_test_frame': is_distractor_test_frame})

        return self.processing(data)
