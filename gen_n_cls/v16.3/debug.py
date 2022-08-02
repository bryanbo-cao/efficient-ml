import glob

# path = '/home/brcao/Data/datasets/coco_datasets_v16.3/coco_minitrain_25k/labels_n_cls_3'
path = '/home/brcao/Data/datasets/coco_datasets_v16.3/coco_minitrain_25k/labels'
id_set = set()
for i, path_ in enumerate(glob.glob(path + '/val2017/*.txt')):
    # print(path_)
    with open(path_, 'r') as f:
        lines = f.readlines()
        for line_i, line in enumerate(lines):
            # cls_ls = line[1:-2].replace(',', '').split(' ')
            # cls_ls = cls_ls.split(' ')
            print(line[0])
            id_set.add(line[0])

print(id_set)
