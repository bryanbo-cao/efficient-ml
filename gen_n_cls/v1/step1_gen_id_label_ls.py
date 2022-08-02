'''
Usage:
python3 step1_gen_id_label_ls.py -drp /home/brcao/Repos/datasets/coco
python3 step1_gen_id_label_ls.py -drp /home/brcao/Repos/datasets/coco_minitrain_25k
python3 step1_gen_id_label_ls.py -dtp /home/brcao/Data/datasets/coco_minitrain_25k
python3 step1_gen_id_label_ls.py -drp /home/brcao/Data/datasets/coco_datasets_v1/coco_minitrain_25k
'''

import os
import cv2
import glob
import argparse
from collections import defaultdict


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

        self.label_folder_dict = defaultdict()
        for data_type in self.data_types:
            self.label_folder_dict[data_type] = self.label_root_path + '/{}2017'.format(data_type)

        self.img_path_to_save = './vis_save'
        if not os.path.exists(self.img_path_to_save):
            os.makedirs(self.img_path_to_save)
        self.label_root_path = self.dataset_root_path + '/labels'
        self.label_folder_dict = defaultdict()
        for data_type in self.data_types:
            self.label_folder_dict[data_type] = self.label_root_path + '/{}2017'.format(data_type)

        self.cat_label_path = self.dataset_root_path + '/coco-labels-2014_2017.txt'
        self.cat_id_label_path = self.dataset_root_path + '/coco-id-labels-2014_2017.txt'


if __name__ == '__main__':
    C = Config()

    if os.path.exists(C.cat_id_label_path):
        cmd = 'rm {}'.format(C.cat_id_label_path)
        os.system(cmd)
    f_write = open(C.cat_id_label_path, 'w')
    with open(C.cat_label_path) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line_to_write = str(i) + ',' + line.replace('\n', '') + '\n'
            f_write.write(line_to_write)
            print(line_to_write, 'written!')
