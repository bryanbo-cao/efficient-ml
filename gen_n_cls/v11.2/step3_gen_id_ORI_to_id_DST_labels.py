'''
Usage:
python3 step3_gen_id_ORI_to_id_DST_labels.py -drp /home/brcao/Repos/datasets/coco
python3 step3_gen_id_ORI_to_id_DST_labels.py -drp /home/brcao/Repos/datasets/coco_minitrain_25k
python3 step3_gen_id_ORI_to_id_DST_labels.py -dtp /home/brcao/Data/datasets/coco_minitrain_25k
python3 step3_gen_id_ORI_to_id_DST_labels.py -drp /home/brcao/Data/datasets/coco_datasets_v11.2/coco_minitrain_25k
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
        self.cat_id_to_name_dict = defaultdict()

        self.cat_id_label_path = self.dataset_root_path + '/coco-id-labels-2014_2017.txt'
        self.cat_id_label_f = open(self.cat_id_label_path, 'r')
        self.cat_id_label_lines = self.cat_id_label_f.readlines()

        self.cls_ls_path = self.dataset_root_path + '/cls_ls.txt'
        self.cls_ls = []
        self.cls_ls_f = open(self.cls_ls_path, 'r')
        self.id_DST_to_id_ORI_labels_dict_path = self.dataset_root_path + '/id_DST_to_id_ORI_labels.json'
        self.id_DST_to_id_ORI_labels_dict = defaultdict()
        self.id_ORI_to_id_DST_labels_dict_path = self.dataset_root_path + '/id_ORI_to_id_DST_labels.json'
        self.id_ORI_to_id_DST_labels_dict = defaultdict()


if __name__ == '__main__':
    C = Config()

    lines = C.cls_ls_f.readlines()
    n_cls_80_ls = lines[-1][1:-2].replace(',', '').split(' ')
    # n_cls_80_ls = list(n_cls_80_ls)
    print('n_cls_80_ls: ', n_cls_80_ls)
    print('type(n_cls_80_ls): ', type(n_cls_80_ls))
    print('len(n_cls_80_ls): ', len(n_cls_80_ls))
    for i, id_cls_DST in enumerate(n_cls_80_ls):
        label = C.cat_id_label_lines[int(id_cls_DST)].split(',')[-1][:-1]
        print('label: ', label)
        C.id_ORI_to_id_DST_labels_dict[id_cls_DST] = [str(i), label]
        C.id_DST_to_id_ORI_labels_dict[str(i)] = [id_cls_DST, label]

    print('C.id_ORI_to_id_DST_labels_dict: ', C.id_ORI_to_id_DST_labels_dict)
    with open(C.id_ORI_to_id_DST_labels_dict_path, 'w') as f:
        json.dump(C.id_ORI_to_id_DST_labels_dict, f, cls=NpEncoder)
    print(C.id_ORI_to_id_DST_labels_dict_path, 'written!')

    print('C.id_DST_to_id_ORI_labels_dict: ', C.id_DST_to_id_ORI_labels_dict)
    with open(C.id_DST_to_id_ORI_labels_dict_path, 'w') as f:
        json.dump(C.id_DST_to_id_ORI_labels_dict, f, cls=NpEncoder)
    print(C.id_DST_to_id_ORI_labels_dict_path, 'written!')
