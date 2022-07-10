'''
Usage:
python3 step2_gen_var_cls_ls.py -drp /home/brcao/Repos/datasets/coco
python3 step2_gen_var_cls_ls.py -drp /home/brcao/Repos/datasets/coco_minitrain_25k
python3 step2_gen_var_cls_ls.py -dtp /home/brcao/Data/datasets/coco_minitrain_25k
python3 step2_gen_var_cls_ls.py -drp /home/brcao/Data/datasets/coco_datasets_v2/coco_minitrain_25k
'''

import os
import cv2
import glob
import argparse
from collections import defaultdict
import copy

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

        self.cls_ls_path = self.dataset_root_path + '/cls_ls.txt'
        self.cls_ls = []
        self.cls_ls_f = open(self.cls_ls_path, 'w')

        self.cls_name_ls_path = self.dataset_root_path + '/cls_name_ls.txt'
        self.cls_name_ls = []
        self.cls_name_ls_f = open(self.cls_name_ls_path, 'w')

if __name__ == '__main__':
    C = Config()

    # [0, 1, 2, 15, 22]: ['person', 'bicycle', 'car', 'cat', 'zebra']
    C.cls_ls.append([0]) # ['person']
    C.cls_ls.append([0, 22]) # ['person', 'zebra']
    C.cls_ls.append([0, 22, 15]) # ['person', 'zebra', 'cat']
    C.cls_ls.append([0, 22, 15, 1]) # ['person', 'zebra', 'cat', 'bicycle']
    C.cls_ls.append([0, 22, 15, 1, 2]) # ['person', 'zebra', 'cat', 'bicycle', 'car']
    visited = copy.deepcopy(C.cls_ls[-1])

    C.cls_name_ls.append(['person'])
    C.cls_name_ls.append(['person', 'zebra'])
    C.cls_name_ls.append(['person', 'zebra', 'cat'])
    C.cls_name_ls.append(['person', 'zebra', 'cat', 'bicycle'])
    C.cls_name_ls.append(['person', 'zebra', 'cat', 'bicycle', 'car'])
    visited_names = copy.deepcopy(C.cls_name_ls[-1])

    for n_cls in range(1, 6):
        n_cls_path = C.dataset_root_path + '/labels_n_cls_{}'.format(n_cls)
        print(n_cls_path) # e.g. /home/brcao/Repos/datasets/labels/coco_n_cls_1
        if not os.path.exists(n_cls_path): os.makedirs(n_cls_path)

    for n_cls in range(10, 81, 10):
        n_cls_path = C.dataset_root_path + '/labels_n_cls_{}'.format(n_cls)
        print(n_cls_path) # e.g. /home/brcao/Repos/datasets/labels/coco_n_cls_1
        if not os.path.exists(n_cls_path): os.makedirs(n_cls_path)


    for n_cls in range(80):
        # if n_cls < 5: or n_cls % 10 == 0:
        if True:
            if n_cls < 5:
                new_cls_ls = C.cls_ls[n_cls]; new_cls_name_ls = C.cls_name_ls[n_cls]
            else:
                print('\n ==============================')
                print('\n C.cls_ls: ', C.cls_ls)
                print('\n n_cls: ', n_cls)
                with open(C.cat_label_path) as f:
                    lines = f.readlines()
                    new_cls_ls = copy.deepcopy(C.cls_ls[-1]); new_cls_name_ls = copy.deepcopy(C.cls_name_ls[-1])
                    cls_added = len(new_cls_ls); cls_name_added = len(new_cls_name_ls)
                    for line_i, line in enumerate(lines):
                        if line_i not in visited:
                            new_cls_ls.append(line_i); new_cls_name_ls.append(line[:-1])
                            visited.append(line_i)
                            cls_added += 1
                            if cls_added >= n_cls:
                                break

                    C.cls_ls.append(new_cls_ls); C.cls_name_ls.append(new_cls_name_ls)
            line_to_write = str(new_cls_ls) + '\n'; line_name_to_write = str(new_cls_name_ls) + '\n'
            print('\n\n line_to_write: ', line_to_write); print('\n\n line_name_to_write: ', line_name_to_write)
            print('\n len(new_cls_ls): ', len(new_cls_ls)); print('\n len(new_cls_name_ls): ', len(new_cls_name_ls))
            C.cls_ls_f.write(line_to_write); C.cls_name_ls_f.write(line_name_to_write)

'''

























'''
