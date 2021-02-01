# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
import torch
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from torchvision.ops import nms

def gather_sequence_info(detections):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    seq_info = {
        "detections": detections,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
    }
    return seq_info


def create_detections(detection_mat, frame_idx):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[7:]
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(detections, **kwargs):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """

    min_confidence = kwargs['min_confidence']
    nms_max_overlap = kwargs['nms_max_overlap']
    max_cosine_distance = kwargs['max_cosine_distance']
    nn_budget = kwargs['nn_budget']
    max_age = kwargs['max_age']
    n_init = kwargs['n_init']
    seq_info = gather_sequence_info(detections)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age, n_init=n_init)
    results = []

    def frame_callback(vis, frame_idx):
        #print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(seq_info["detections"], frame_idx)
        detections = [d for d in detections if d.confidence >= min_confidence]
        # Run non-maxima suppression.
        if nms_max_overlap >= 0:
            boxes = np.array([d.to_tlbr() for d in detections])
            scores = np.array([d.confidence for d in detections])
            
            nms_keep = nms(torch.from_numpy(boxes),
                                torch.from_numpy(scores),
                                iou_threshold=nms_max_overlap).numpy()
            
            detections = [detections[i] for i in nms_keep]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Store results.
        for track in tracker.tracks:
            #only output the detection results, no prediction
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue
            bbox = track.last_tlwh
            feature = track.last_feature
            confidence = track.last_confidence
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], confidence] + feature.tolist())

    # Run tracker.
    visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)
    
    return results