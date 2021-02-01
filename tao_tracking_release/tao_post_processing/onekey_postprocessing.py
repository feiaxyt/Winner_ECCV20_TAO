import argparse
import itertools
import json
import logging
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_json_for_eval import create_json
from track_concat import track_concat
from associate_track import track_associate


def track_and_save(onnx_path1, onnx_path2, output, threshold_ass):

    onnx_detections1 = np.load(onnx_path1)
    onnx_detections2 = np.load(onnx_path2)
    #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, category, feature
    
    det_concat = track_concat(onnx_detections1, onnx_detections2)
    
    det_associate = track_associate(det_concat, threshold_ass)
     
    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(output, det_associate[:, :8])


def track_and_save_star(args):
    track_and_save(*args)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--onnx_results1',
                        type=Path,
                        required=True,
                        help='results that use the first det')
    parser.add_argument('--onnx_results2',
                        type=Path,
                        required=True,
                        help='results that use the second det')
    parser.add_argument('--annotations',
                        type=Path,
                        required=True,
                        help='Annotations json')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory, where a results.json will be output.')

    parser.add_argument('--threshold-ass', type=float, default=0.7, help='threshold for track association')
    
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    def get_output_path(video):
        return args.output_dir / (video + '.npy')

    with open(args.annotations, 'r') as f:
        groundtruth = json.load(f)
    
    videos = [x['name'] for x in groundtruth['videos']]

    tasks = []
    for video in tqdm(videos):
        onnx_path1 = args.onnx_results1 / (video + '.npy')
        onnx_path2 = args.onnx_results2 / (video + '.npy')
        output = get_output_path(video)
        tasks.append((onnx_path1, onnx_path2, output, args.threshold_ass))

    if args.workers > 0:
        pool = Pool(args.workers)
        list(
            tqdm(pool.imap_unordered(track_and_save_star, tasks),
                 total=len(tasks),
                 desc='Tracking'))
    else:
        for task in tqdm(tasks):
            track_and_save(*task)
    print(f'Finished')
    
    if 'train' in str(args.annotations):
        output = args.output_dir / 'train_results.json'
    elif 'val' in str(args.annotations):
        output = args.output_dir / 'val_results.json'
    elif 'test'in str(args.annotations):
        output = args.output_dir / 'test_results.json'
    else:
        print('wrong annotations')

    create_json(args.output_dir, groundtruth, output)


if __name__ == "__main__":
    main()
