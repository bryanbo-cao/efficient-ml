scp -r /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k_ORI
scp -r /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k/images /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k/images_ORI
scp -r /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k/labels /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k/labels_ORI
scp -r /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k/train2017.txt /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k/train2017_ORI.txt
scp -r /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k/val2017.txt /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k/val2017_ORI.txt
python3 step0_cvt_scale_coco_bbox.py -drp /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k
python3 step1_gen_id_label_ls.py -drp /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k
python3 step2_gen_var_cls_ls.py -drp /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k
python3 step3_gen_id_ORI_to_id_DST_labels.py -drp /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k
python3 step4_gen_var_labels.py -drp /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k
python3 step5_copy_labels.py -drp /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k
python3 step6_gen_yamls.py -rp /home/brcao/Repos/yolov5 -drp /home/brcao/Data/datasets/coco_datasets_v6.2/coco_minitrain_25k
