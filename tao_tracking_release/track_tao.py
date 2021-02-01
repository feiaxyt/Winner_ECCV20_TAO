import argparse
import itertools
import json
import logging
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import deep_sort_app
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tao_post_processing.create_json_for_eval import create_json


def track_and_save(det_path, output, save_feature, sort_kwargs):

    detections = np.load(det_path)
   
    #['image_id', 'category','bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'] features 512
    
    categories = np.unique(detections[:, 1].astype(int))

    id_gen = itertools.count(1)
    unique_track_ids = defaultdict(lambda: next(id_gen))

    all_results = []
    #th = 32 ** 2
    for category in categories:
        mask = detections[:, 1].astype(int) == category
        det = detections[mask]
        
        results = deep_sort_app.run(det, **sort_kwargs) #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, feature
        if len(results) == 0:
            continue

        results = np.array(results).reshape(len(results), -1)
        track_ids = np.array([unique_track_ids[(x, category)] for x in results[:, 1]])
        category_res = np.ones((results.shape[0]), dtype=np.float32) * category
        results_save = np.hstack((results[:, 0:1], track_ids[:, np.newaxis], results[:, 2:7], category_res[:, np.newaxis]))
        if save_feature:
            results_save = np.hstack((results_save, results[:, 7:]))
        all_results.append(results_save)

    all_results = np.concatenate(all_results, axis=0) #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, category
    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(output, all_results)


def track_and_save_star(args):
    track_and_save(*args)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detections-dir',
                        type=Path,
                        required=True,
                        help='Results directory with features')
    parser.add_argument('--annotations',
                        type=Path,
                        required=True,
                        help='Annotations json')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory, where a results.json will be output'))
    parser.add_argument('--save-feature',
                        action='store_true',
                        help='whether to save features for post processing')
    #deepsort args
    parser.add_argument('--min_confidence',
                        default=-1,
                        type=float,
                        help='Float or "none".')
    parser.add_argument('--nms_max_overlap', type=float, default=-1)
    parser.add_argument('--max_cosine_distance', default=0.4, type=float)
    parser.add_argument('--nn_budget', default=70, type=int)
    parser.add_argument('--max_age', default=12, type=int)
    parser.add_argument('--n_init', default=1, type=int)
    
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()
    args.nms_max_overlap = (-float('inf') if args.nms_max_overlap == 'none'
                            else float(args.nms_max_overlap))

    args.output_dir.mkdir(exist_ok=True, parents=True)

    def get_output_path(video):
        return args.output_dir / (video + '.npy')

    with open(args.annotations, 'r') as f:
        groundtruth = json.load(f)
    videos = [x['name'] for x in groundtruth['videos']]
    video_paths = {}
    for video in tqdm(videos, desc='Collecting paths'):
        output = get_output_path(video)
        if output.exists():
            print(f'{output} already exists, skipping...')
            continue
        detection_paths = args.detections_dir / video / 'det.npy'
        
        assert detection_paths.exists()
        video_paths[video] = detection_paths

    if not video_paths:
        print(f'Nothing to do! Exiting.')
        return
    print(f'Found {len(video_paths)} videos to track.')

    tasks = []
    for video, path in tqdm(video_paths.items()):
        output = get_output_path(video)
        tasks.append((path, output, args.save_feature, {
                          'min_confidence': args.min_confidence,
                          'nms_max_overlap': args.nms_max_overlap,
                          'max_cosine_distance': args.max_cosine_distance,
                          'nn_budget': args.nn_budget,
                          'max_age': args.max_age,
                          'n_init': args.n_init,
                      }))

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
