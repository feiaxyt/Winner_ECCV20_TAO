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
from torchvision.ops import nms
from tqdm import tqdm
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def save_detections(images, detection_dir, output, score_threshold, nms_thresh, num_det):
    items = {'image_id': [], 'bb_left': [], 'bb_top': [], 'bb_width': [], 'bb_height': [], 'conf': [], 'category': []}
    sorted(images, key=lambda x: x['id'])
    for image in images:
        pkl_path = (detection_dir / image['file_name']).with_suffix('.pkl')
        if not pkl_path.exists():
            pkl_path = pkl_path.with_suffix('.jpeg')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)['instances']
        instances = {
            'scores': np.array(data['scores']),
            'pred_classes': np.array(data['pred_classes']),
            'pred_boxes': np.array(data['pred_boxes'])
        }
        if len(instances['scores']) == 0:
            print(str(pkl_path) + ' is empty')
        if score_threshold > 0:
            valid = instances['scores'] > score_threshold
            for x in ('scores', 'pred_boxes', 'pred_classes'):
                instances[x] = instances[x][valid]
                
        categories = sorted({ x for x in instances['pred_classes'] })
        all_instances = {'scores': [], 'pred_boxes': [], 'pred_classes': []}
        
        for category in categories:
            in_class = instances['pred_classes'] == category
            class_instances = {
                k: instances[k][in_class]
                for k in ('scores', 'pred_boxes', 'pred_classes')
            }
            if nms_thresh >= 0:
                nms_keep = nms(torch.from_numpy(class_instances['pred_boxes']),
                               torch.from_numpy(class_instances['scores']),
                               iou_threshold=nms_thresh).numpy()
                class_instances = {
                    'scores': class_instances['scores'][nms_keep],
                    'pred_boxes': class_instances['pred_boxes'][nms_keep],
                    'pred_classes': class_instances['pred_classes'][nms_keep]
                }
            for x in ('scores', 'pred_boxes', 'pred_classes'):
                all_instances[x].extend(class_instances[x])
                
        for x in ('scores', 'pred_boxes', 'pred_classes'):
                all_instances[x] = np.array(all_instances[x])
        
        
        sorted_index = np.argsort(all_instances['scores'])[::-1]
        for x in ('scores', 'pred_boxes', 'pred_classes'):
            all_instances[x] = all_instances[x][sorted_index[:num_det]]
            
        for (score, bbox, cat) in zip(all_instances['scores'], all_instances['pred_boxes'], all_instances['pred_classes']):
            items['image_id'].append(image['id'])
            items['bb_left'].append(bbox[0])
            items['bb_top'].append(bbox[1])
            items['bb_width'].append(bbox[2]-bbox[0])
            items['bb_height'].append(bbox[3]-bbox[1])
            items['conf'].append(score)
            items['category'].append(cat)
    det_dfs = pd.DataFrame(items)
    
    output.parent.mkdir(exist_ok=True, parents=True)
    columns = ['image_id','bb_left','bb_top', 'bb_width', 'bb_height', 'conf', 'category']
    det_dfs.to_csv(output, header=False, index=False, columns=columns)
    
def save_detections_star(args):
    save_detections(*args)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detections-dir',
                        type=Path,
                        required=True,
                        help='detection directory with pickle files')
    parser.add_argument('--annotations',
                        type=Path,
                        required=True,
                        help='Annotations json')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory'))

    parser.add_argument('--output-file',
                        type=str,
                        default='new_det.txt',
                        help='name of txt output file')
    
    parser.add_argument('--score-threshold',
                        default=-1,
                        help='threshold to filter detections')

    parser.add_argument('--nms-thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--det-num', type=int, default=300, help='number of detections to save')
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()
    
    args.score_threshold = (-float('inf') if args.score_threshold == 'none'
                            else float(args.score_threshold))

    args.output_dir.mkdir(exist_ok=True, parents=True)

    def get_output_path(video):
        return args.output_dir / video / args.output_file

    with open(args.annotations, 'r') as f:
        groundtruth = json.load(f)
    
    videos = [x['name'] for x in groundtruth['videos']]
    video2images = defaultdict(list)
    for image in groundtruth['images']:
        video2images[image['video']].append(image)

    tasks = []
    for video in tqdm(videos):
        output = get_output_path(video)
        tasks.append((video2images[video], args.detections_dir, output, args.score_threshold, args.nms_thresh, args.det_num))

    if args.workers > 0:
        pool = Pool(args.workers)
        list(
            tqdm(pool.imap_unordered(save_detections_star, tasks),
                 total=len(tasks),
                 desc='Save Detections'))
    else:
        for task in tqdm(tasks):
            save_detections(*task)
    print('Finished')

if __name__ == "__main__":
    main()
