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

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b.T)

def noverlap(period1, period2):
    s1 = period1[0]
    e1 = period1[-1]
    s2 = period2[0]
    e2 = period2[-1]
        
    if (0 < s2 - e1) or (0 < s1 - e2):
        return True
     
    return False

def reid_similarity(det1, det2):
    feat1 = det1[:, 8:]
    feat2 = det2[:, 8:]
    avg_feat1 = np.mean(feat1, axis=0)
    avg_feat2 = np.mean(feat2, axis=0)
    return cosine_similarity(avg_feat1, avg_feat2)

def associate(det, threshold):
    processed_track_list = []
    match = {}
    tids = np.unique(det[:, 1])
    for i in range(len(tids) - 1):
        if i in processed_track_list:
            continue
        for j in range(i+1, len(tids)):
            if j in processed_track_list:
               continue
            det_i = det[det[:, 1] == tids[i]]
            det_j = det[det[:, 1] == tids[j]]
            image_ids_i = det_i[:, 0]
            image_ids_j = det_j[:, 0]
            if noverlap(image_ids_i, image_ids_j):
                similarity = reid_similarity(det_i, det_j)
                if similarity > threshold:
                    match[tids[j]] =  tids[i]
                    processed_track_list.append(j)
    if len(match) != 0:
        results = []
        for tid in tids:
            sub_det = det[det[:, 1] == tid]
            if tid in match:
                sub_det[:, 1] = match[tid]
            results.append(sub_det)
        det = np.vstack(results)
    return det

def track_associate(detections, threshold):
    categories = np.unique(detections[:, 7].astype(int))

    all_results = []
    
    for category in categories:
        mask = detections[:, 7].astype(int) == category
        det = detections[mask].copy()

        results = associate(det, threshold)

        all_results.append(results)

    all_results = np.concatenate(all_results, axis=0)
    return all_results            

def track_and_save(det_path, output, threshold):

    detections = np.load(det_path)
   
    #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, category, feature
    results = track_associate(detections, threshold)
    
    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(output, results[:, :8])


def track_and_save_star(args):
    track_and_save(*args)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results-dir',
                        type=Path,
                        required=True,
                        help='track results')
    parser.add_argument('--annotations',
                        type=Path,
                        required=True,
                        help='Annotations json')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=('Output directory, where a results.json will be output, as well '
              'as a .npz file for each video, containing a boxes array of '
              'size (num_boxes, 6), of the format [x0, y0, x1, y1, class, '
              'score, box_index, track_id], where box_index maps into the '
              'pickle files'))

    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for track association')
    
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()

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
        
        result_path = args.results_dir / (video + '.npy')
        
        assert result_path, (
            f'No result file at {result_path}!')
        video_paths[video] = result_path

    if not video_paths:
        print(f'Nothing to do! Exiting.')
        return

    print(f'Found {len(video_paths)} videos to track.')

    tasks = []
    for video, path in tqdm(video_paths.items()):
        output = get_output_path(video)
        tasks.append((path, output, args.threshold))

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

if __name__ == "__main__":
    main()
