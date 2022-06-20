import os
import cv2
import glob
import argparse
from collections import defaultdict
import copy
import json

# TypeError: Object of <some_code> is not JSON serializable
# Ref: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# -----------------------------------
#  Configurations for One Experiment
# -----------------------------------
class Config:
    def __init__(self):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--draw_bbox', type=bool, default=False)
        parser.add_argument('-v', '--visualize_bbox', type=bool, default=False)
        # parser.add_argument('-s', '--scale', type=int, default=2)
        self.args = parser.parse_args()
        self.args_dict = vars(self.args)

        self.dataset_root_path = '/home/brcao/Repos/datasets/coco_minitrain_8k'
        self.img_root_path = self.dataset_root_path + '/images'
        self.label_root_path = self.dataset_root_path + '/labels'
        self.data_types = ['train', 'val']
        self.img_folder_dict = defaultdict()

        self.img_folder_dict['1o1'] = defaultdict()
        for data_type in self.data_types:
            self.img_folder_dict['1o1'][data_type] = self.img_root_path + '/{}2017'.format(data_type)

        self.img_id_ls = []

if __name__ == '__main__':
    C = Config()

    for scaled_str, img_folder in C.img_folder_dict.items():
        # print(scaled_str, img_folder)
        # for data_type in C.data_types:
        if True:
            data_type = 'train'
            img_path = img_folder[data_type]
            print('\n\n img_path: ', img_path)
            # e.g img_path:  /home/brcao/Repos/datasets/coco_minitrain_8k/images/train2017

            for img_path in glob.glob(img_path + '/*.jpg'):
                print(img_path)
                # e.g. /home/brcao/Repos/datasets/coco_minitrain_8k/images/train2017/000000443453.jpg
                img_id = img_path[img_path.rindex('/') + 1 : img_path.index('.jpg')]
                print(img_id)
                # e.g. 000000443453
                C.img_id_ls.append(img_id)

        assert len(C.img_id_ls) == 8000
        print(len(C.img_id_ls)) # 8000
