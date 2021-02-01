# coding=utf-8
import os
import cv2
import sys
import pdb
import subprocess
import multiprocessing

import inference_utils

import common


def single_process(index, task, gpu):

    print(("任务%d处理%d张图片" % (index, len(task))))

    # 写文件
    filename = inference_utils.dump_testfile(task, index)

    out_str = subprocess.check_output(["python", file, "--gpuid=%s" % str(gpu), "--img_list=%s" % filename, "--out_dir=%s" % out_dir, "--batch_size=%d" % batch_size])

    print(("任务%d处理完毕！" % (index)))


if "__main__" == __name__:

    gpu_list = [1,2,2,3,3,4,4,5,5,6,6,7,7]
    file = "tools/multi_process_inference/inference.py"
    img_dir = '/home/songbai.xb/workspace/projects/TAO/data/TAO/frames/train/'
    out_dir = './tmp/file/train_nonms_tta/'
    batch_size = 1

    # 解析dir
    img_list = common.load_filepaths(img_dir, suffix=('.jpg', '.png', '.jpeg'), recursive=True)
    #names = demo_utils.parse_testfile(testfile)
    print(f"总共{len(img_list)}张图片")

    # 分任务
    task_num = len(gpu_list)
    tasks = inference_utils.chunks(img_list, task_num)

    # 创建进程
    processes=list()
    for idx, (task, gpu) in enumerate(zip(tasks, gpu_list)):
        processes.append(multiprocessing.Process(target=single_process,args=(idx, task, gpu)))

    for process in processes:
        process.start()

    for process in processes:
        process.join()