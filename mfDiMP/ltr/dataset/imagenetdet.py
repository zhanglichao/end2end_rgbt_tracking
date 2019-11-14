import os
from .base_dataset import BaseDataset
from ltr.data.image_loader import jpeg4py_loader
# import xml.etree.ElementTree as ET
# import pickle
import torch
# from collections import OrderedDict
from collections import OrderedDict
import xml.etree.ElementTree as ET
import json
from ltr.admin.environment import env_settings


# import nltk
# from nltk.corpus import wordnet


    # def get_target_to_image_ratio(self):
    #     return (self.anno[0, 2:4].prod() / (self.image_size.prod())).sqrt()

def construct_det_sequence(data_dict):
    return DETSequence(data_dict['img_folder'], data_dict['img_filename'], data_dict['class_name'], data_dict['target_bb'],
                       data_dict['image_size'])


def get_target_to_image_ratio(seq):
    target_bb = torch.Tensor(seq['target_bb'])
    img_sz = torch.Tensor(seq['image_size'])
    return (target_bb[2:4].prod() / (img_sz.prod())).sqrt()

class ImagenetDETSeq(BaseDataset):
    """

    Args:
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, max_target_area=1):
        super().__init__(root, image_loader)
        raise NotImplementedError
        root = env_settings().imagenetdet_dir if root is None else root
        cache_file = os.path.join(root, 'cache.json')
        if os.path.isfile(cache_file):
            with open(cache_file, 'r') as f:
                self.sequence_list = json.load(f)

        else:
            self.sequence_list = self._process_anno(root)

            with open(cache_file, 'w') as f:
                json.dump(self.sequence_list, f)

        self.sequence_list = [x for x in self.sequence_list if get_target_to_image_ratio(x) < max_target_area]

        self.class_list = self._get_class_list()

    def _get_class_list(self):
        # sequence list is a
        class_list = []
        for x in self.sequence_list:
            class_list.append(x['class_name'])
        class_list = list(set(class_list))
        class_list.sort()
        return class_list

    def is_video_sequence(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'imagenetdet'

    def has_class_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)


    def get_sequences_in_class(self, class_name):
        return None

    def get_sequence_info(self, seq_id):
        anno = torch.Tensor(self.sequence_list[seq_id]['target_bb']).view(1, 4)
        visible = torch.Tensor([1])
        valid = torch.Tensor([1])
        return  {'anno': anno, 'valid': valid, 'visible': visible}

    def _get_anno(self, seq_id):
        anno = self.sequence_list[seq_id]['target_bb']
        return torch.Tensor(anno).view(1, 4)

    def _get_frames(self, seq_id):
        frame_path = os.path.join(self.root, 'ILSVRC2014_DET_train', self.sequence_list[seq_id]['img_folder'],
                                  self.sequence_list[seq_id]['img_filename'] + '.JPEG')
        return self.image_loader(frame_path)

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self._get_anno(seq_id)

        anno_frames = [anno.clone()[0, :] for _ in frame_ids]

        object_meta = OrderedDict({'object_class': self.sequence_list[seq_id]['class_name'],
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def _process_anno(self, root):
        base_anno_path = os.path.join(root, 'ILSVRC2014_DET_bbox_train')

        all_sequences = []
        for set in sorted(os.listdir(base_anno_path)):
            for vid in sorted(os.listdir(os.path.join(base_anno_path, set))):
                try:
                    anno_file = os.path.join(base_anno_path, set, vid)

                    anno = ET.parse(anno_file)

                    img_folder = anno.find('folder').text
                    img_filename = anno.find('filename').text

                    image_size = [int(anno.find('size/width').text), int(anno.find('size/height').text)]

                    objects = ET.ElementTree(file=anno_file).findall('object')

                    for target in objects:
                        class_name = target.find('name').text
                        x1 = int(target.find('bndbox/xmin').text)
                        y1 = int(target.find('bndbox/ymin').text)
                        x2 = int(target.find('bndbox/xmax').text)
                        y2 = int(target.find('bndbox/ymax').text)

                        target_bb = [x1, y1, x2 - x1, y2 - y1]

                        all_sequences.append({'img_folder': img_folder, 'img_filename': img_filename, 'class_name': class_name,
                                              'target_bb': target_bb, 'image_size': image_size})
                except:
                    print('Not detection image')
        return all_sequences
