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

def track_concat(detections1, detections2):
    id_gen = itertools.count(1)
    unique_track_ids = defaultdict(lambda: next(id_gen))

    all_detections = []
    tids = np.unique(detections1[:, 1].astype(int))
    for tid in tids:
        det = detections1[detections1[:, 1].astype(int)==tid].copy()
        det[:,1] = unique_track_ids[(tid, 0)]
        all_detections.append(det)

    tids = np.unique(detections2[:,1].astype(int))
    for tid in tids:
        det = detections2[detections2[:,1].astype(int)==tid].copy()
        det[:, 1] = unique_track_ids[(tid, 1)]
        all_detections.append(det)

    detections = np.vstack(all_detections)
    return detections

def track_and_save(det_path1, det_path2,  output):

    detections1 = np.load(det_path1)
    detections2 = np.load(det_path2)
    #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, category, feature

    detections = track_concat(detections1, detections2)
    
    output.parent.mkdir(exist_ok=True, parents=True)
    np.save(output, detections)
    
def track_and_save_star(args):
    track_and_save(*args)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results-dir1',
                        type=Path,
                        required=True,
                        help='the first results')
    parser.add_argument('--results-dir2',
                        type=Path,
                        required=True,
                        help='the second results')
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

    
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    #npz_dir = dir_path(args.output_dir)

    def get_output_path(video):
        return args.output_dir / (video + '.npy')

    with open(args.annotations, 'r') as f:
        groundtruth = json.load(f)
    
    videos = [x['name'] for x in groundtruth['videos']]

    tasks = []
    for video in tqdm(videos):
        path1 = args.results_dir1 / (video + '.npy')
        path2 = args.results_dir2 / (video + '.npy')
        output = get_output_path(video)
        tasks.append((path1, path2, output))

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
