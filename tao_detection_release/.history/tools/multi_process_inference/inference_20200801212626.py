import argparse
import os
import os.path as osp
import shutil
import tempfile
import pdb
import numpy as np
import pickle

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import tao_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset, FilesDataset
from mmdet.models import build_detector

from mmdet.core import build_assigner

import pickle
import common
from inference_utils import *



def single_gpu_test(model, data_loader, show=False, cfg=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', type=str, default='/home/songbai.xb/workspace/research/detection/BalancedGroupSoftmax/configs/tao/gs_htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_lvis.py', help='test config file path')
    parser.add_argument('--checkpoint', type=str, default='/home/songbai.xb/workspace/research/detection/BalancedGroupSoftmax/data/pretrained_models/gs_htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_lvis.pth', help='checkpoint file')
    parser.add_argument('--img_dir', type=str, default='', help='inference img dir')
    parser.add_argument('--img_list', type=str, default='', help='inference img dir')
    parser.add_argument('--out_dir', type=str, default='./tmp/file/val/', help='output dir')
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="batch_size default 32")
    parser.add_argument("-g",'--gpuid', type=str, default='0', help='visible gpu ids')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def reweight_cls(model, tauuu):

    if tauuu == 0:
        return model

    model_dict = model.state_dict()

    def pnorm(weights, tau):
        normB = torch.norm(weights, 2, 1)
        ws = weights.clone()

        for i in range(0, weights.shape[0]):
            ws[i] = ws[i] / torch.pow(normB[i], tau)

        return ws

    reweight_set = ['bbox_head.fc_cls.weight']
    tau = tauuu
    for k in reweight_set:
        weight = model_dict[k]  # ([1231, 1024])
        weight = pnorm(weight, tau)
        model_dict[k].copy_(weight)
        print('Reweight param {:<30} with tau={}'.format(k, tau))

    return model



def save_in_tao_format(mmdet_res, results_path):
    '''
    Args:
        mmdet_res (list[array]):  一张图的结果
    '''

    boxes_decoded = []
    scores_decoded = []
    classes_decoded = []
    for i, dets in enumerate(mmdet_res):
        for det in dets:
            boxes_decoded.append([det[0], det[1], det[2], det[3]])
            scores_decoded.append(det[4])
            classes_decoded.append(i)
    
    predictions_decoded = {}
    predictions_decoded["instances"] = {
        "pred_boxes": boxes_decoded,
        "scores": scores_decoded,
        "pred_classes": classes_decoded,
    }
    
    with open(results_path, 'wb') as f:
        pickle.dump(predictions_decoded, f)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    img_dir = args.img_dir
    out_dir = args.out_dir
    batch_size = args.batch_size

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    if args.img_dir != '':
        file_list = common.load_filepaths(args.img_dir, suffix=('.jpg', '.png', '.jpeg'), recursive=True)
    elif  args.img_list != '':
        file_list = parse_testfile(args.img_list)
    else:
        raise "Both img_dir and img_list is empty."

    dataset = FilesDataset(file_list, cfg.test_pipeline)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=batch_size,
        workers_per_gpu=batch_size,
        dist=distributed,
        shuffle=False)
    
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = reweight_cls(model, args.tau).cuda()

    model = MMDataParallel(model, device_ids=[0])

    model.eval()
    count = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # bbox_results, segm_results
            results = model(return_loss=False, rescale=True, **data)
        
        # batch
        #for result  in results:
        #    file_path = file_list[count]
        #    save_name = file_path.replace('/home/songbai.xb/workspace/projects/TAO/data/TAO/frames/val/', '')
        #    save_path = os.path.join(out_dir, save_name)
        #    common.makedirs(os.path.dirname(save_path))
        #    save_in_tao_format(result, save_path)
        #    count += 1
        file_path = file_list[i]
        save_name = file_path.replace('/home/songbai.xb/workspace/projects/TAO/data/TAO/frames/val/', '') 
        save_name = save_name.replace('.jpg', '.pkl')
        save_path = os.path.join(out_dir, save_name)
        common.makedirs(os.path.dirname(save_path))
        save_in_tao_format(results[0], save_path)
        
if __name__ == '__main__':
    main()
