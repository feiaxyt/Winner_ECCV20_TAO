

# htc_x101
#CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/baselines/htc_x101_64x4d_fpn_20e_16gpu_lvis.py \
#./data/pretrained_models/htc_x101_64x4d_fpn_20e_16gpu_lvis.pth  7 \
# --out results/lvis_val/htc_x101_result.pkl --eval bbox


# faster_rcnn
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/tao/faster_rcnn_x101_64x4d_fpn_1x_lvis.py \
#./data/pretrained_models/faster_rcnn_x101_64x4d_fpn_1x_lvis.pth  6 \
# --out results/lvis_val/faster_rcnn_result.pkl --eval bbox


# cascade_rcnn_x101
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/bags/gs_cascade_rcnn_x101_64x4d_fpn_1x_lvis.py \
./data/pretrained_models/gs_cascade_rcnn_x101_64x4d_fpn_1x_lvis.pth  7 \
 --out results/lvis_val/cascade_x101_result.pkl --eval bbox






