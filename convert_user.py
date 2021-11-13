import json
import os
import os.path as osp
from glob import glob
from PIL import Image
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, ConcatDataset, Dataset


SRC_DATASET_DIR = '/opt/ml/input/data/USER'      # FIXME
DST_DATASET_DIR = '/opt/ml/input/data/USER_UFO'  # FIXME

NUM_WORKERS = 32  # FIXME

IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'PNG', 'png']

LANGUAGE_MAP = {
    'KO': 'ko',
    'EN': 'en',
    'None': None
}

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann        
        
class CustomDataset(Dataset):
    def __init__(self, root, label_dir, copy_images_to=None):

        data = read_json(osp.join(root,label_dir))
        self.root = root 
        self.sample_names = [name for name in data['images'].keys() if name.split('.')[1] in IMAGE_EXTENSIONS];
        self.sample_infos = data['images']
        self.copy_images_to = copy_images_to

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        image_fname = self.sample_names[idx]
        sample_info = self.sample_infos[image_fname]
        
        image = Image.open(osp.join(self.root,'images',image_fname))
        img_w, img_h = image.size

        if self.copy_images_to:
            maybe_mkdir(self.copy_images_to)
            shutil.copy(osp.join(self.root,'images',self.sample_names[idx]),osp.join(self.copy_images_to,self.sample_names[idx]))

        for k,v in sample_info['words'].items() :
            if v['language'] is not None :
                v['language'][0] = v['language'][0].lower()
            
        license_tag = dict(usability=True, public=True, commercial=True, type='CC-BY-SA',
                           holder=None)
        sample_info_ufo = dict(img_h=img_h, img_w=img_w, words=sample_info['words'], tags=None,
                               license_tag=license_tag)

        return image_fname, sample_info_ufo

def main():
    dst_image_dir = osp.join(DST_DATASET_DIR, 'images')

    # set json dir 
    label_name = "new_mask.json"
    custom = CustomDataset(SRC_DATASET_DIR,label_name,copy_images_to=dst_image_dir)

    anno = dict(images=dict())
    with tqdm(total=len(custom)) as pbar:
        for batch in DataLoader(custom, num_workers=NUM_WORKERS, collate_fn=lambda x: x):
            image_fname, sample_info = batch[0]
            anno['images'][image_fname] = sample_info
            pbar.update(1)

    ufo_dir = osp.join(DST_DATASET_DIR, 'ufo')
    maybe_mkdir(ufo_dir)
    with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
        json.dump(anno, f, indent=4)


if __name__ == '__main__' :
    main()
