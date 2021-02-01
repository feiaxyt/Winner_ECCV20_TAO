import argparse
import itertools
import json
import logging
import pickle
import sys
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
from natsort import natsorted
from torchvision.ops import nms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from tao.utils.parallel.fixed_gpu_pool import FixedGpuPool
import time
from numpy import pad
from reid_pytorch.reid_extractor import ReID_Inference
# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

DET_COL_NAMES = ('image_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'category')

class BoundingBoxDataset(Dataset):
    """
    Class used to process detections. Given a DataFrame (det_df) with detections of a MOT sequence, it returns
    the image patch corresponding to the detection's bounding box coordinates
    """
    def __init__(self, det_df, pad_ = True, pad_mode = 'mean',  output_size = (224, 224)):
        self.det_df = det_df
        self.output_size = output_size
        self.pad = pad_
        self.curr_image_path = None
        self.curr_img = None
        self.pad_mode = pad_mode
        self.mean = np.asarray([123.675,116.280,103.530])
        self.std = np.asarray([57.0,57.0,57.0])

    def __len__(self):
        return self.det_df.shape[0]

    def __getitem__(self, ix):
        row = self.det_df.iloc[ix]
        # Load this bounding box' frame img, in case we haven't done it yet
        if row['image_path'] != self.curr_image_path:
            self.curr_img = cv2.imread(row['image_path'])
            if self.curr_img is None:
                self.curr_img = cv2.imread(row['image_path'].replace('jpg', 'jpeg'))
            self.curr_image_path = row['image_path']

        frame_img = self.curr_img
        frame_height = frame_img.shape[0]
        frame_width = frame_img.shape[1]
        # Crop the bounding box, and pad it if necessary to
        bb_img = frame_img[int(max(0, row['bb_top'])): int(max(0, row['bb_bot'])),
                   int(max(0, row['bb_left'])): int(max(0, row['bb_right']))]
        if self.pad:
            x_height_pad = np.abs(row['bb_top'] - max(row['bb_top'], 0)).astype(int)
            y_height_pad = np.abs(row['bb_bot'] - min(row['bb_bot'], frame_height)).astype(int)

            x_width_pad = np.abs(row['bb_left'] - max(row['bb_left'], 0)).astype(int)
            y_width_pad = np.abs(row['bb_right'] - min(row['bb_right'], frame_width)).astype(int)

            bb_img = pad(bb_img, ((x_height_pad, y_height_pad), (x_width_pad, y_width_pad), (0, 0)), mode=self.pad_mode)
        
        bb_img = cv2.resize(bb_img, self.output_size)
        
        bb_img = bb_img[:, :, ::-1]
        bb_img = (bb_img-self.mean)/self.std
        
        bb_img = np.transpose(bb_img, (2, 0, 1)).astype(np.float32)
        return bb_img


def init_model(init_args, context):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(context['gpu'])

def infer(kwargs, context):
    images = kwargs['images']
    image_path = kwargs['image_path']
    det_path = kwargs['det_path']
    output = kwargs['output']
    model_file = kwargs['model_file']
    model = ReID_Inference(model_file)
    det_df = pd.read_csv(det_path, header=None)
    
    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    imageid2filename = {x['id']: x['file_name'] for x in images}
    
    det_df['bb_bot'] = (det_df['bb_top'] + det_df['bb_height']).values
    det_df['bb_right'] = (det_df['bb_left'] + det_df['bb_width']).values
    func = lambda x: str(image_path / imageid2filename[int(x)])
    det_df['image_path'] = det_df['image_id'].apply(func)
    conds = (det_df['bb_width'] > 1) & (det_df['bb_height'] > 1)
    conds = conds & (det_df['bb_right'] > 1) & (det_df['bb_bot'] > 1)
    det_df = det_df[conds]

    bbox_dataset = BoundingBoxDataset(det_df)
    bbox_loader = DataLoader(bbox_dataset, batch_size=500, pin_memory=False, num_workers=4)
    #Feed all bboxes to the CNN to obtain node and reid embeddings
    
    print(f"Computing embeddings for {len(bbox_dataset)} detections")
    #start_time = time.time()
    features = []
    with torch.no_grad():
        for bboxes in bbox_loader:
            bboxes = bboxes.numpy()
            feature = model(bboxes)
            features.append(feature)
            #print(time.time() - start_time, image_path)
    
    features = np.concatenate(features, axis=0)

    data = np.hstack((det_df[['image_id', 'category','bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']].values.astype(np.float32),features))

    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(str(output), data)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detections-dir',
                        type=Path,
                        required=True,
                        help='Results directory with txt detection files')
    parser.add_argument('--annotations',
                        type=Path,
                        required=True,
                        help='Annotations json')
    parser.add_argument(
        '--image-dir',
        type=Path,
        required=True,
        help=('image directory'))
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))
    
    parser.add_argument('--detection-file',
                        type=str,
                        default='new_det.txt',
                        help='name of txt detection file')
    parser.add_argument('--model-file',
                        type=str,
                        default='reid_pytorch/reid1.onnx',
                        help='name of the reid model file')
    parser.add_argument('--gpus', default=[0], nargs='+', type=int)
    args = parser.parse_args()
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    def get_output_path(video):
        return args.output_dir / video / 'det.npy'
    
    def get_image_path(video):
        return args.image_dir / video

    with open(args.annotations, 'r') as f:
        groundtruth = json.load(f)

    videos = [x['name'] for x in groundtruth['videos']]

    video2images = defaultdict(list)
    for image in groundtruth['images']:
        video2images[image['video']].append(image)
    detection_paths = {}   
    for video in tqdm(videos, desc='Collecting paths'):
        output = get_output_path(video)
        if output.exists():
            print(f'{output} already exists, skipping...')
            continue
        detection_path = args.detections_dir / (os.path.join(video, args.detection_file))
        
        assert detection_path.exists(), (
            f'No detections dir at {detection_path}!')
        
        detection_paths[video] = detection_path

    if not detection_paths:
        print(f'Nothing to do! Exiting.')
        return
    print(f'Found {len(detection_paths)} videos to track.')

    infer_tasks = []
    for video, det_path in tqdm(detection_paths.items()):
        output = get_output_path(video)
        infer_tasks.append({'images': video2images[video], 'image_path': args.image_dir, 
                            'det_path': det_path, 'output':output, 'model_file': args.model_file})
    
    init_args = []
    if len(args.gpus) == 1:
        context = {'gpu': args.gpus[0]}
        init_model(init_args, context)
        for task in tqdm(infer_tasks,
                         mininterval=1,
                         desc='Running generation',
                         dynamic_ncols=True):
            infer(task, context)
    else:
        pool = FixedGpuPool(
            args.gpus, initializer=init_model, initargs=init_args)
        list(
            tqdm(pool.imap_unordered(infer, infer_tasks),
                 total=len(infer_tasks),
                 mininterval=10,
                 desc='Running generation',
                 dynamic_ncols=True))
    print(f'Finished')

if __name__ == "__main__":
    main()
