'''
Usage:
python3 step4_gen_var_labels.py -drp /home/brcao/Repos/datasets/coco
python3 step4_gen_var_labels.py -drp /home/brcao/Repos/datasets/coco_minitrain_25k
python3 step4_gen_var_labels.py -drp /home/brcao/Data/datasets/coco_minitrain_25k
python3 step4_gen_var_labels.py -drp /home/brcao/Data/datasets/coco_datasets_v10.3/coco_minitrain_25k
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

        self.cls_ls_path = self.dataset_root_path + '/cls_ls.txt'

        self.id_ORI_to_id_DST_labels_dict_path = self.dataset_root_path + '/id_ORI_to_id_DST_labels.json'
        with open(self.id_ORI_to_id_DST_labels_dict_path, 'r') as f:
            self.id_ORI_to_id_DST_labels_dict = json.load(f)
            print(self.id_ORI_to_id_DST_labels_dict_path, 'loaded!')

        self.id_DST_to_id_ORI_labels_dict_path = self.dataset_root_path + '/id_DST_to_id_ORI_labels.json'
        with open(self.id_DST_to_id_ORI_labels_dict_path, 'r') as f:
            self.id_DST_to_id_ORI_labels_dict = json.load(f)
            print(self.id_DST_to_id_ORI_labels_dict_path, 'loaded!')


if __name__ == '__main__':
    C = Config()

    with open(C.cls_ls_path) as f:
        lines = f.readlines()
        # print('\n lines: ', lines)

        # ------------------------------------------
        #  Iterate Through n_cls lists from 1 to 80
        # ------------------------------------------
        for line_i, line in enumerate(lines):
            cls_ls = line[1:-2].replace(',', '').split(' ')
            n_cls = len(cls_ls)
            id_ORI_ls, id_DST_ls = [], []

            if n_cls < 6 or n_cls % 10 == 0:
                print('\n===============================================')
                print('\n\n line_i: ', line_i, ', cls_ls: ', cls_ls, ', n_cls: ', n_cls)
                n_cls_path = C.dataset_root_path + '/labels_n_cls_{}'.format(n_cls)
                print(n_cls_path) # e.g. /home/brcao/Repos/datasets/labels/coco_n_cls_1
                if not os.path.exists(n_cls_path): os.makedirs(n_cls_path)
                for data_type in C.data_types:
                    # debug
                    # if data_type == 'val': continue
                    label_folder_ORI = C.label_folder_dict[data_type]

                    # labels_*
                    label_folder_DST = n_cls_path + '/{}2017'.format(data_type)
                    # print('\n\n label_folder_DST: ', label_folder_DST)
                    if os.path.exists(label_folder_DST):
                        cmd = 'rm -r {}'.format(label_folder_DST); os.system(cmd); print(cmd)
                    os.makedirs(label_folder_DST)

                    # *.txt
                    label_txt_DST = C.dataset_root_path + '/{}2017_n_cls_{}.txt'.format(data_type, n_cls)
                    # print('\n\n label_txt_DST: ', label_txt_DST)
                    label_txt_DST_f = open(label_txt_DST, 'a')

                    # Make a copy of the original one
                    # label_txt = C.dataset_root_path + '/{}2017.txt'.format(data_type)
                    # label_txt_ORI = C.dataset_root_path + '/{}2017_ORI.txt'.format(data_type)
                    # print('\n\n label_txt: ', label_txt, ', label_txt_ORI: ', label_txt_ORI)
                    # copy_cmd = 'scp -r {} {}'.format(label_txt, label_txt_ORI)
                    # print(copy_cmd)
                    # os.system(copy_cmd)

                    # ------------------------------------------
                    #  Iterate Through each img label text file
                    # ------------------------------------------
                    # print('\n-----------------------------------------------')
                    for label_path_ORI in glob.glob(label_folder_ORI + '/*.txt'):
                        img_id = label_path_ORI[label_path_ORI.rindex('/') + 1:label_path_ORI.index('.txt')]
                        # print('\n\n img_id: ', img_id)

                        # labels_*
                        label_path_ORI_f = open(label_path_ORI, 'r')
                        label_path_ORI_lines = label_path_ORI_f.readlines()
                        # ------------------------------------------
                        #  Iterate Through each label within an img
                        # ------------------------------------------
                        # print('\n -------------------------*.txt-----------------------------')
                        # print('\n label_path_ORI: ', label_path_ORI)
                        # print('\n label_path_ORI_lines: ', label_path_ORI_lines)

                        for label_path_ORI_line_i, label_path_ORI_line in enumerate(label_path_ORI_lines):
                            # print('label_path_ORI_line: ', label_path_ORI_line)
                            cat_id_ORI = label_path_ORI_line.split(' ')[0]
                            # print('n_cls: ', n_cls, ', cat_id_ORI: ', cat_id_ORI)

                            # id_ORI_ls.add(cat_id_ORI)

                            if cat_id_ORI in cls_ls:
                                label_path_DST = label_folder_DST + '/' + img_id + '.txt'
                                # print('\n\n label_path_DST: ', label_path_DST)

                                label_path_DST_f = open(label_path_DST, 'a')

                                cat_id_DST = C.id_ORI_to_id_DST_labels_dict[cat_id_ORI][0]
                                if cat_id_ORI not in id_ORI_ls: id_ORI_ls.append(cat_id_ORI)
                                if cat_id_DST not in id_DST_ls: id_DST_ls.append(cat_id_DST)
                                # print('n_cls: ', n_cls, ', cat_id_DST: ', cat_id_DST)
                                # if cat_id_DST == str(n_cls - 1): print('n_cls: ', n_cls, ', cat_id_DST: ', cat_id_DST)
                                # print('label_path_ORI_line: ', label_path_ORI_line)
                                label_path_DST_line_ls = label_path_ORI_line_ls = label_path_ORI_line.split(' ')
                                label_path_DST_line_ls[0] = cat_id_DST
                                label_path_DST_line = ' '.join(label_path_DST_line_ls)
                                # print('label_path_DST_line: ', label_path_DST_line)
                                label_path_DST_f.write(label_path_DST_line)

                                if cat_id_ORI != '0':
                                    print('label_path_DST_line: ', label_path_DST_line)
                                # DEBUG:
                                # if int(cat_id_DST) > n_cls:
                                #     print('Error')

                        # *.txt
                        str_to_write = './images/{}2017/{}.jpg\n'.format(data_type, img_id)
                        # print('str_to_write: ', str_to_write)
                        label_txt_DST_f.write(str_to_write)

                print('\n\n\n ================ cat_id_ORI: ', cat_id_ORI, ',  cat_id_DST: ', cat_id_DST, ', n_cls: ', n_cls, '====================')
                print('\n\n\n ================ (id_ORI_ls): ', (id_ORI_ls), ',  (id_DST_ls): ', (id_DST_ls), '====================')
                print('\n\n\n ================ len(id_ORI_ls): ', len(id_ORI_ls), ',  len(id_DST_ls): ', len(id_DST_ls), '====================')
