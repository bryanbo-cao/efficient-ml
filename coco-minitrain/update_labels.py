'''
Usage:
python3 update_labels.py -mtp /home/brcao/Repos/datasets/coco_minitrain_8k
'''

import os
import cv2
import glob
import argparse
from collections import defaultdict
import copy
import json

# -----------------------------------
#  Configurations for One Experiment
# -----------------------------------
class Config:
    def __init__(self):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        parser = argparse.ArgumentParser()
        parser.add_argument('-mtp', '--minitrain_path', type=str, default='coco_minitrain')
        self.args = parser.parse_args()
        self.args_dict = vars(self.args)

        self.dataset_root_path = self.args.minitrain_path # edit
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

    print('---------------------------')

    txt_path = C.dataset_root_path + '/train2017.txt'
    print(txt_path)
    txt_path_ORI = C.dataset_root_path + '/train2017_ORI.txt'
    if not os.path.exists(txt_path_ORI):
        cmd = 'scp {} {}'.format(txt_path, txt_path_ORI); os.system(cmd); print(cmd)
        cmd = 'rm {}'.format(txt_path); os.system(cmd); print(cmd)
        cmd = 'touch {}'.format(txt_path); os.system(cmd); print(cmd)

    label_root_path = C.dataset_root_path + '/labels'
    print(label_root_path)
    label_root_path_ORI = C.dataset_root_path + '/labels_ORI'
    label_train_path = label_root_path + '/train2017'
    if not os.path.exists(label_root_path_ORI):
        cmd = 'scp -r {} {}'.format(label_root_path, label_root_path_ORI); os.system(cmd); print(cmd)
        cmd = 'rm -r {}'.format(label_train_path); os.system(cmd); print(cmd)
        os.makedirs(label_train_path)

    label_val_path = label_root_path + '/val2017'
    if os.path.exists(label_val_path):
        label_val_path_ORI = label_root_path_ORI + '/val2017'
        cmd = 'scp -r {} {}'.format(label_val_path_ORI, label_val_path); os.system(cmd); print(cmd)

    print('---------------------------')

    txt_file = open(txt_path, 'w')
    for img_id in C.img_id_ls:
        # ----------------------
        #  Update train2017.txt
        # ----------------------
        line_to_write = './images/train2017/{}.jpg\n'.format(img_id)
        txt_file.write(line_to_write)
        print(line_to_write, 'written!')

        # ---------------------------
        #  Update labels/train2017/*
        # ---------------------------
        label_path = label_root_path + '/train2017/' + img_id + '.txt'
        label_path_ORI = label_root_path_ORI + '/train2017/' + img_id + '.txt'
        cmd = 'scp {} {}'.format(label_path_ORI, label_path); os.system(cmd); print(print)
        print(cmd)

    # Clean intermediate folders
    cmd = 'rm -r {}'.format(label_root_path_ORI); os.system(cmd); print(cmd)
    cmd = 'rm -r {}'.format(txt_path_ORI); os.system(cmd); print(cmd)
