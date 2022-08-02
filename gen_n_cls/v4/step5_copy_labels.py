'''
Usage:
python3 step5_copy_labels.py -drp /home/brcao/Repos/datasets/coco
python3 step5_copy_labels.py -drp /home/brcao/Repos/datasets/coco_minitrain_25k
python3 step5_copy_labels.py -drp /home/brcao/Data/datasets/coco_minitrain_25k
python3 step5_copy_labels.py -drp /home/brcao/Data/datasets/coco_datasets_v4/coco_minitrain_25k
'''

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
        parser.add_argument('-drp', '--dataset_root_path', type=str, default='coco') # root path of the dataset
        parser.add_argument('-b', '--draw_bbox', type=bool, default=False)
        parser.add_argument('-v', '--visualize_bbox', type=bool, default=False)
        # parser.add_argument('-s', '--scale', type=int, default=2)
        self.args = parser.parse_args()
        self.args_dict = vars(self.args)

        self.dataset_root_path = self.args.dataset_root_path
        self.img_root_path = self.dataset_root_path + '/images'
        self.label_root_path = self.dataset_root_path + '/labels'
        self.data_types = ['train', 'val']
        self.img_folder_dict = defaultdict()

        self.img_folder_dict['1o1'] = defaultdict()
        for data_type in self.data_types:
            self.img_folder_dict['1o1'][data_type] = self.img_root_path + '/{}2017'.format(data_type)

        self.scale_ls = [] # [1/2, 1/4, 1/8, 1/16]
        self.scale_str_ls = [] # ['1o2', '1o4', '1o8', '1o16']
        for i, scale in enumerate(self.scale_ls):
            self.scaled_dataset_root_path = self.dataset_root_path + '_' + self.scale_str_ls[i]
            if not os.path.exists(self.scaled_dataset_root_path): os.makedirs(self.scaled_dataset_root_path)
            self.scaled_img_root_path = self.scaled_dataset_root_path + '/images'
            if not os.path.exists(self.scaled_img_root_path): os.makedirs(self.scaled_img_root_path)

            self.img_folder_dict[self.scale_str_ls[i]] = defaultdict()
            for data_type in self.data_types:
                self.img_folder_dict[self.scale_str_ls[i]][data_type] = self.scaled_img_root_path + '/{}2017'.format(data_type)
                if not os.path.exists(self.img_folder_dict[self.scale_str_ls[i]][data_type]): os.makedirs(self.img_folder_dict[self.scale_str_ls[i]][data_type])


if __name__ == '__main__':
    C = Config()

    for scaled_str, img_folder in C.img_folder_dict.items():
        print(scaled_str, img_folder)
        for data_type in C.data_types:
            img_path = img_folder[data_type]
            dataset_root_path = img_path[:img_path.index('/images')]
            print('\n\n dataset_root_path: ', dataset_root_path)

            copy_cmd = 'scp -r {}/labels_* {}'.format(C.dataset_root_path, dataset_root_path)
            print(copy_cmd)
            os.system(copy_cmd)

            copy_cmd = 'scp -r {}/*.txt {}'.format(C.dataset_root_path, dataset_root_path)
            print(copy_cmd)
            os.system(copy_cmd)

            copy_cmd = 'scp -r {}/*.json {}'.format(C.dataset_root_path, dataset_root_path)
            print(copy_cmd)
            os.system(copy_cmd)
