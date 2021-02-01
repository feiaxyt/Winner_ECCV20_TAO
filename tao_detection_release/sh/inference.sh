

CUDA_VISIBLE_DEVICES=4,5,6,7  tools/dist_inference.sh configs/tao/faster_rcnn_x101_64x4d_fpn_1x_tao.py \
 ./data/pretrained_models/faster_rcnn_x101_64x4d_fpn_1x_lvis.pth 4 '/home/songbai.xb/workspace/projects/TAO/data/TAO/frames/val/'  'tmp/file/val'
 