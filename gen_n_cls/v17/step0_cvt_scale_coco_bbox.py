'''
Usage:
python3 step0_cvt_scale_coco_bbox.py -drp /home/brcao/Repos/datasets/coco
python3 step0_cvt_scale_coco_bbox.py -drp /home/brcao/Repos/datasets/coco_minitrain_25k
python3 step0_cvt_scale_coco_bbox.py -dtp /home/brcao/Data/datasets/coco_minitrain_25k
python3 step0_cvt_scale_coco_bbox.py -drp /home/brcao/Data/datasets/coco_datasets_v17/coco_minitrain_25k
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
        if not os.path.exists(self.cat_label_path):
            cmd = 'git clone https://github.com/amikelive/coco-labels'
            os.system(cmd)
            cmd = 'pwd'
            os.system(cmd)
            cmd = 'scp coco-labels/coco-labels-2014_2017.txt {}'.format(self.dataset_root_path)
            os.system(cmd)
            cmd = 'rm -rf coco-labels'
            os.system(cmd)
        self.cat_id_to_name_dict = defaultdict()
        with open(self.cat_label_path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.cat_id_to_name_dict[i] = line.replace('\n', '')

        print(self.cat_id_to_name_dict)

        # -------
        #  Color
        # -------
        self.color_ls = ['crimson', 'lime green', 'royal blue', 'chocolate', 'purple', 'lemon']
        self.color_dict = {
            'crimson': (60,20,220),
            'lime green': (50,205,50),
            'royal blue': (225,105,65),
            'chocolate': (30,105,210),
            'purple': (128,0,128),
            'lemon': (0,247,255),
            'dark orange': (255,140,0),
            'dark green': (0,100,0),
            'dark turquoise': (0,206,209),
            'dodger blue': (30,144,255)
        }


if __name__ == '__main__':
    C = Config()

    # Each Data Type
    for data_type in C.data_types:
        print('C.img_folder_dict[1o1][data_type]: ', C.img_folder_dict['1o1'][data_type])

        # Each Image
        for img_path in glob.glob(C.img_folder_dict['1o1'][data_type] + '/*.jpg'):
            img_id = img_path[img_path.rindex('/'):img_path.index('.jpg')]
            # print(img_path)

            img = cv2.imread(img_path)
            r, c = img.shape[0], img.shape[1]
            dim_ORI = (c, r)

            # Draw bbx >>>
            if C.args.draw_bbox:
                label_path = C.label_folder_dict[data_type] + '/' + img_id + '.txt'
                print('label_path: ', label_path)
                if os.path.exists(label_path):
                    with open(label_path) as f:
                        lines = f.readlines()
                        for line in lines:
                            # print(line)
                            line = line.split(' ')
                            label_id = int(line[0])
                            x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                            # print(img.shape)
                            # top_left = (int((y - w / 2) * img.shape[0]), int((x - h / 2) * img.shape[1]))
                            # bottom_right = (int((y + w / 2) * img.shape[0]), int((x + h / 2) * img.shape[1]))

                            color = C.color_dict[C.color_ls[label_id % len(C.color_ls)]]
                            # point = (int(y * r), int(x * c))
                            point = (int(x * c), int(y * r))
                            img = cv2.circle(img, point, 2, color, 2)

                            top_left = (int((x - w / 2) * c), int((y - h / 2) * r))
                            bottom_right = (int((x + w / 2) * c), int((y + h / 2) * r))
                            img = cv2.rectangle(img, top_left, bottom_right, color, 2)

                            img = cv2.putText(img, C.cat_id_to_name_dict[label_id], (top_left[0] - 5, top_left[1] - 5), \
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            # Draw bbox <<<

            # vis >>>
            if C.args.visualize_bbox: cv2.imshow('img', img); cv2.waitKey(0)

            # Save images with different scales
            for i, scale_str in enumerate(C.scale_str_ls):

                dim_resized = (int(c * C.scale_ls[i]), int(r * C.scale_ls[i])) # (width, height)
                img_resized = cv2.resize(img, dim_resized)
                if C.args.visualize_bbox: cv2.imshow('img_resized', img_resized); cv2.waitKey(0)

                img_res = cv2.resize(img_resized, dim_ORI)
                if C.args.visualize_bbox: cv2.imshow('img_res', img_res); cv2.waitKey(0)

                # save
                img_path_to_save = C.img_folder_dict[scale_str][data_type] + '/' + img_id + '.jpg'
                cv2.imwrite(img_path_to_save, img_res)
                # print(img_path_to_save, 'saved!')

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
