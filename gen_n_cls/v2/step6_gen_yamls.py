'''
Usage:
python3 step6_gen_yamls.py -rp /home/brcao/Repos/yolov5 -drp /home/brcao/Repos/datasets/coco
python3 step6_gen_yamls.py -rp /home/brcao/Repos/yolov5 -drp /home/brcao/Repos/datasets/coco_minitrain_25k
python3 step6_gen_yamls.py -rp /home/brcao/Repos/yolov5 -dtp /home/brcao/Data/datasets/coco_minitrain_25k
python3 step6_gen_yamls.py -rp /home/brcao/Repos/yolov5 -drp /home/brcao/Data/datasets/coco_datasets_v2/coco_minitrain_25k
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
        parser.add_argument('-drp', '--dataset_root_path', type=str, default='coco') # root path of dataset
        parser.add_argument('-rp', '--repo_root_path', type=str, default='yolov5') # root path of repo
	# parser.add_argument('-s', '--scale', type=int, default=2)
        self.args = parser.parse_args()
        self.args_dict = vars(self.args)

        self.dataset_root_path = self.args.dataset_root_path
        self.dataset = self.dataset_root_path[self.dataset_root_path.rindex('/') + 1 : ]
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
        self.cls_ls_f = open(self.cls_ls_path, 'r')

        self.cls_name_ls_path = self.dataset_root_path + '/cls_name_ls.txt'
        self.cls_name_ls = []
        self.cls_name_ls_f = open(self.cls_name_ls_path, 'r')

        self.data_yaml_folder = self.args.repo_root_path + '/data'
        self.data_yaml_f = open(self.data_yaml_folder + '/coco.yaml', 'r')

        self.model_folder = self.args.repo_root_path + '/models'
        self.model_f = None # open(self.model_folder + '/yolov5n.yaml', 'r')

        self.model_ls = ['yolov5n', 'yolov5_p1p2p3_4', 'yolov5_p1p2p3p4_4', 'yolov5_p1p2p3p4p5_4']

if __name__ == '__main__':
    C = Config()
    # ========
    #  Models
    # ========
    for n_cls, name_ls in enumerate(C.cls_name_ls_f.readlines()):
        n_cls += 1
        print('\n\n n_cls: ', n_cls, ', name_ls: ', name_ls)
        for model in C.model_ls:
            model_yaml_f = open(C.model_folder + '/' + model + '.yaml', 'r')
            new_model_yaml_f = open(C.model_folder + '/' + model + '_n_cls_' + str(n_cls) + '.yaml', 'w')
            lines = model_yaml_f.readlines()
            for line_i, line in enumerate(lines):
                if line.startswith('nc'):
                    new_line = 'nc: {}\n'.format(str(n_cls))
                else: new_line = line
                new_model_yaml_f.write(new_line)

    # ======
    #  Data
    # ======
    C.scale_str_ls.append('1o1')
    for scale_str in C.scale_str_ls:
        print('\n\n scale_str: ', scale_str)
        C.cls_name_ls_path = C.dataset_root_path + '/cls_name_ls.txt'
        C.cls_name_ls_f = open(C.cls_name_ls_path, 'r')
        for n_cls, name_ls in enumerate(C.cls_name_ls_f.readlines()):
            n_cls += 1
            if n_cls < 6 or n_cls % 10 == 0:
                print('\n\n n_cls: ', n_cls, ', name_ls: ', name_ls)
                if scale_str == '1o1':
                    dataset_root_path = C.dataset_root_path
                    # new_data_yaml_f = open(C.data_yaml_folder + '/' + C.dataset + '_n_cls_' + str(n_cls) + '.yaml', 'w')
                    new_data_yaml_f = open(C.data_yaml_folder + '/' + C.dataset + '_n_cls_' + str(n_cls) + '_v2.yaml', 'w') # edit
                else:
                    dataset_root_path = C.dataset_root_path + '_' + scale_str
                    # new_data_yaml_f = open(C.data_yaml_folder + '/' + C.dataset + '_' + scale_str + '_n_cls_' + str(n_cls) + '.yaml', 'w')
                    new_data_yaml_f = open(C.data_yaml_folder + '/' + C.dataset + '_' + scale_str + '_n_cls_' + str(n_cls) + '_v2.yaml', 'w')
                print('\n\n dataset_root_path: ', dataset_root_path)

                C.data_yaml_f = open(C.data_yaml_folder + '/coco.yaml', 'r')
                lines = C.data_yaml_f.readlines()
                for line_i, line in enumerate(lines):
                    if line.startswith('path'):
                        new_line = 'path: {}\n'.format(dataset_root_path)
                    elif line.startswith('nc'):
                        new_line = 'nc: {}\n'.format(n_cls)
                    elif line.startswith('names'):
                        new_line = 'names: {}\n'.format(name_ls)
                    elif '# Download script/URL (optional)' in line:
                        break
                    else:
                        new_line = line
                    new_data_yaml_f.write(new_line)
                    print('\n\n ', new_line)
                    if line.startswith('names'): break
