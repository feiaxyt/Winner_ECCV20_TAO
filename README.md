# Code for the winner of TAO 2020

## Note: 

This repo is **NOT** a fully reinplementation of our submitted result([mAP@27.46](https://motchallenge.net/results/TAO_Challenge/)). 

Instead, two models for appearance modeling are included, together with the open-source [BAGS](https://github.com/FishYuLi/BalancedGroupSoftmax) model and the full set of code for inference. 
With this code, you can achieve around mAP@23 with TAO test set (based on our estimation).

# Start from here:

Please find instructions for the detection model in [tao_detection_release/README.md](tao_detection_release/README.md) and the tracking model in [tao_tracking_release/README.md](tao_tracking_release/README.md)

# Result file：

The result for the TAO validation set can be dowloaded in [TAO_val](https://drive.google.com/file/d/1NGLcQ40ci2MfbJK34hq37mv9iTOXV2sh/view?usp=sharing).

# License

Apache 2.0

# Citation

If you find this code useful, please cite our arxiv tech report:

> @article{Du_2020_TAO,  
  author    = {Fei Du and
               Bo Xu and
               Jiasheng Tang and
               Yuqi Zhang and
               Fan Wang and
               Hao Li},  
  title     = {1st Place Solution to {ECCV-TAO-2020:} Detect and Represent Any Object for Tracking},  
  journal   = {arXiv preprint arXiv: 2101.08040},  
  year      = {2021},  
  url       = {https://arxiv.org/abs/2101.08040}  
}
