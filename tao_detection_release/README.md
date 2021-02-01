
# 1. Environment setup

The requirements are exactly the same as [BAGS](https://github.com/FishYuLi/BalancedGroupSoftmax). We tested on on the following settings:
- python 3.7
- cuda 10.0
- pytorch 1.4.1+cu100
- torchvision 0.5.0+cu100
- mmcv 0.2.10

For convience, we add our inferecne code for TAO detection-output extraction on BAGS. Note that the main code and models of this folder are the same as BAGS.

Thanks to the great work provided by Yu et al.

We are considering to release a better detection model with BAGS and DetectoRS in the future, or you can train your own model based on our description in the arxiv report.

# 2. Prepare Dataset and Model

- [Download](https://drive.google.com/drive/folders/1UsCscmh7F6KOya1K7R2vTT5KswsIFVXd)  intermediate files of BAGS (`label2binlabel.pt, pred_slice_with0.pt, valsplit.pkl`). Put them under `./data/lvis_v0.5/`.
- Generate tao_img_list.txt. Put them under `./data/tao/`.
- [Download](https://drive.google.com/file/d/1QkfpYYAHgEym8KVg8a7zclSHe5LGssjH/view) the BAGS model. Put them under `./work_dirs/bags/`.

**After all the abovementioned steps, the folder data should be like this:**

```
- data
    - tao
        - frames/
        - annotations/
        - train_img_list.txt
        - val_img_list.txt
        - test_img_list.txt
    - lvis_v0.5
        - annotations
            - label2binlabel.pt
            - pred_slice_with0.pt
            - valsplit.pt
- work_dirs
    - bags
        - gs_htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_lvis.pth
```

The train_img_list.txt data format is as follows, you can easily get it from the JSON provided by TAO:
```
train/YFCC100M/v_8b6283255797fc7e94f3a93947a2803/frame1171.jpg
train/YFCC100M/v_8b6283255797fc7e94f3a93947a2803/frame1201.jpg
train/YFCC100M/v_8b6283255797fc7e94f3a93947a2803/frame1231.jpg
train/YFCC100M/v_8b6283255797fc7e94f3a93947a2803/frame1261.jpg
train/YFCC100M/v_8b6283255797fc7e94f3a93947a2803/frame1291.jpg
...
```

# 3. Inference
- You can change the GPU resources in the `tools/multi_process_inference/multi_inference.py` code based on the number of GPUs.
- The detection results are saved in the `./results` folder.
- Please note that the model backbone large, therefore the inference takes a long time (7h+ in total).
- To save time, you can download the detection results directly [TAO_alldet.zip](https://drive.google.com/file/d/1cH3aYvCqzdT2PS27rWrC2ECj1d_H3HWy/view?usp=sharing)

```
cd tao_detection_release
mkdir results

python tools/multi_process_inference/multi_inference.py train
python tools/multi_process_inference/multi_inference.py val 
python tools/multi_process_inference/multi_inference.py test
```
