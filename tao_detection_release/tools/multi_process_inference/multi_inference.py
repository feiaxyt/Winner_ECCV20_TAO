# coding=utf-8
import os
import cv2
import sys
import tempfile
import math
import subprocess
import multiprocessing
import pdb



def dump_testfile(task, index, tmp_dir):
    filename = os.path.join(tmp_dir.name, "testfile_%d.txt" % index)
    fp = open(filename, "w+")
    for t in task:
        s = "%s\n" % t
        fp.write(s)
    fp.close()
    return  filename

def parse_textfile(testfile):
    with open(testfile) as fp:
        namelist = [x.strip() for x in fp.readlines()]
    return namelist

# split the arr into N chunks
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def single_process(index, task, gpu, split):

    print(("任务%d处理%d张图片" % (index, len(task))))

    # 写文件
    tmp_dir = tempfile.TemporaryDirectory()
    filename = dump_testfile(task, index, tmp_dir)

    out_str = subprocess.check_output(["python", file, "--gpuid=%s" % str(gpu), "--img_list=%s" % filename, 
            "--out_dir=%s" % out_dir, "--batch_size=%d" % batch_size, "--split=%s" % split])

    print(("任务%d处理完毕！" % (index)))


if "__main__" == __name__:
    import sys
    split = sys.argv[1]
    print(split)

    gpu_list = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
    # gpu_list = [0,0,1,1]
    file = "tools/multi_process_inference/inference.py"
    img_txt = f'./data/tao/{split}_img_list.txt'
    out_dir = f'./results/'
    batch_size = 1

    # 解析dir
    img_list = parse_textfile(img_txt)
    print(f"总共{len(img_list)}张图片")

    # 分任务
    task_num = len(gpu_list)
    tasks = chunks(img_list, task_num)

    # 创建进程
    processes=list()
    for idx, (task, gpu) in enumerate(zip(tasks, gpu_list)):
        processes.append(multiprocessing.Process(target=single_process,args=(idx, task, gpu, split)))

    for process in processes:
        process.start()

    for process in processes:
        process.join()
