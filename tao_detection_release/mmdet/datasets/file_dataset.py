# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: py36_cu10_clone
#     language: python
#     name: py36_cu10_clone
# ---

# +
import glob
from torch.utils.data import Dataset

from mmdet.datasets.pipelines import Compose
from .registry import DATASETS

@DATASETS.register_module
class FilesDataset(Dataset):
    '''只支持test mode
    '''
    
    def __init__(self,
                 files,
                 pipeline,
                 img_prefix=None,
                 anno_path=None,
                 test_mode=True):
        '''
        files (str or list) : file dir or file list
        '''
        if isinstance(files, list):
            if img_prefix is not None:
                img_files = [ os.path.join(img_prefix, file) for file in files ]
            else:
                img_files = files
        else:
            raise('Wrong input files type')
            
        self.img_files = img_files
        self.img_infos =  [dict(filename=img_file) for img_file in img_files]
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
    
    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        
    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info, img_prefix='')
        
        return self.pipeline(results)

    
    
if __name__=='__main__':
    files = '/home/songbai.xb/projects/BaoGuoXia_uniform/datasets/0225-0227/img_all/'
    files_list_txt = '/home/songbai.xb/projects/BaoGuoXia_uniform/datasets/0225-0227//part_anno/part_imgs.txt'
    with open(files_list_txt,'r') as fp:
        files_list = [ line.strip() for line in fp.readlines()]
        
    
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', img_scale=(512, 1024), keep_ratio=False),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    
    dataset = FilesDataset(files_list, test_pipeline, test_mode=True)





