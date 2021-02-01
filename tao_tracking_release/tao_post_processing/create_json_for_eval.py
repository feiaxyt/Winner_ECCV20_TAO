import argparse
import json
import logging
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def remove_1len_track(all_results):
    refined_results = []
    tids = np.unique(all_results[:, 1])
    for tid in tids:
        results = all_results[all_results[:, 1] == tid]
        if results.shape[0] <= 1:
            continue
        refined_results.append(results)
    refined_results = np.concatenate(refined_results, axis=0)
    return refined_results

def create_json(track_result, groundtruth, output_dir):
    # Image without extension -> image id
    image_stem_to_info = {
        x['file_name']: x for x in groundtruth['images']
    }
    video2image = defaultdict(list)
    for x in groundtruth['images']:
        video2image[x['video']].append(x)
    valid_videos = {x['name'] for x in groundtruth['videos']}

    all_annotations = []
    found_predictions = {}
    for video in tqdm(valid_videos):
        video_npy = track_result / f'{video}.npy'
        if not video_npy.exists():
            print(f'Could not find video {video} at {video_npy}')
            continue
        video_result = np.load(video_npy)
        video_result = remove_1len_track(video_result)
        images = video2image[video]
        sorted(images, key=lambda x: x['id'])
        video_found = {}
        for frame in images:
            frame_name = frame['file_name']
            det_frame = video_result[video_result[:, 0] == frame['id']]
            if len(det_frame) != 0:
                video_found[frame_name] = True
            #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, category
            all_annotations.extend([{
                'image_id': frame['id'],
                'video_id':  frame['video_id'],
                'track_id': int(x[1]),
                'bbox': [x[2], x[3], x[4], x[5]],
                'score': x[6],
                'category_id': int(x[7])+1, #if baseline, class + 1, or test and val
            } for x in det_frame])
        if not video_found:
            raise ValueError(f'Found no valid predictions for video {video}')
        found_predictions.update(video_found)
    if not found_predictions:
        raise ValueError('Found no valid predictions!')
    
    with_predictions = set(found_predictions.keys())
    with_labels = set(image_stem_to_info.keys())
    if with_predictions != with_labels:
        missing_videos = {
            x.rsplit('/', 1)[0]
            for x in with_labels - with_predictions
        }
        print(
            f'{len(with_labels - with_predictions)} images from '
            f'{len(missing_videos)} videos did not have predictions!')


    with open(output_dir, 'w') as f:
        json.dump(all_annotations, f)

def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--track-result', required=True, type=Path)
    parser.add_argument('--annotations',
                        type=Path,
                        help='Annotations')
    parser.add_argument('--output-dir', required=True, type=Path)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    with open(args.annotations, 'r') as f:
        groundtruth = json.load(f)
    
    if 'train' in str(args.annotations):
        output = args.output_dir / 'train_results.json'
    elif 'val' in str(args.annotations):
        output = args.output_dir / 'val_results.json'
    elif 'test'in str(args.annotations):
        output = args.output_dir / 'test_results.json'
    else:
        print('wrong annotations')

    create_json(args.track_result, groundtruth, output)
 


if __name__ == "__main__":
    main()
