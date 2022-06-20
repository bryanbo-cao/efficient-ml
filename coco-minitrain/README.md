

Conda Env
```
conda create --name coco
conda activate coco
```

Installing pycocotools
https://github.com/matterport/Mask_RCNN/issues/6
```
git clone https://github.com/pdollar/coco.git
cd PythonAPI
pip3 install Cython --install-option="--no-cython-compile"
python3 setup.py build_ext --inplace
sudo python3 setup.py install
```


```
git clone https://github.com/bryanbo-cao/coco-minitrain
cd src
python3 sample_coco.py --coco_path /home/brcao/Repos/datasets/coco --save_file_name "train2017_mini_8k" --save_format "json" --sample_image_count 8000 --run_count 10
python3 coco_download.py --annotation train2017_mini_8k.json --output train2017_mini_8k
```

```
coco_minitrain_8k/
    annotations/
        train2017.json (renamed from 'train2017_mini_8k.json')
    images/
        train2017/ (renamed from 'train2017_mini_8k')
            *.jpg
        val2017/ (ORI)
            *.jpg
    labels/
        train2017/ (ORI, MDF)
            *.txt
        val2017/ (ORI)
            *.txt
    coco-id-labels-2014_2017.txt
    test-dev2017.txt
    train2017.txt (ORI, MDF)
    val2017.txt (ORI)
```

```ORI``` indicates that the file or directory is from the ```original``` dataset.

```MDF``` indicates that the file or directory needs to be ```modified```.

**TODO**

Iterate through ```coco_minitrain_8k/images/train2017``` and compare it with the original ```coco/images/train2017``` to find the list of removed image IDs in ```removed_img_id_ls```. Update ```coco_minitrain_8k/train2017.txt``` and ```coco_minitrain_8k/labels``` accordingly.

