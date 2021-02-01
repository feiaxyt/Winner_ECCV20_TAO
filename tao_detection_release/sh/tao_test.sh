
# 单gpu
#CUDA_VISIBLE_DEVICES=7 python tools/test_tao.py configs/tao/faster_rcnn_x101_64x4d_fpn_1x_tao.py \
# ./data/pretrained_models/faster_rcnn_x101_64x4d_fpn_1x_lvis.pth \
# --out results/tao_val/fastrcnn_x101_result.pkl --eval bbox

# 多gpu
## faster_rcnn_x101_64x4d_fpn_1x_tao
#CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 tools/dist_test_lvis.sh configs/tao/faster_rcnn_x101_64x4d_fpn_1x_tao.py \
# ./data/pretrained_models/faster_rcnn_x101_64x4d_fpn_1x_lvis.pth 7 \
# --out results/tao_val/fastrcnn_x101_result.pkl --json_out  results/tao_val/0716_result.json --eval bbox

# gs_faster_rcnn_x101
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/tao/gs_faster_rcnn_x101_64x4d_fpn_1x_lvis.py \
./data/pretrained_models/gs_faster_rcnn_x101_64x4d_fpn_1x_lvis_bg8.pth  7 \
 --out results/tao_val/gs_fasterrcnn_x101_result.pkl --eval bbox
 
 
 # gs_cascade_rcnn_x101
#CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/tao/gs_cascade_rcnn_x101_64x4d_fpn_1x_lvis.py \
#./data/pretrained_models/gs_cascade_rcnn_x101_64x4d_fpn_1x_lvis.pth  7 \
# --out results/tao_val/gs_cascade_x101_result.pkl --eval bbox
 
# htc_x101
#CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/tao/htc_x101_64x4d_fpn_20e_16gpu_tao_test.py \
#./data/pretrained_models/htc_x101_64x4d_fpn_20e_16gpu_lvis.pth  7 \
# --out results/tao_val/htc_x101_result.pkl --eval bbox
 
# gs_htc_x101
#CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/tao/gs_htc_x101_64x4d_fpn_20e_16gpu_lvis.py \
#./data/pretrained_models/gs_htc_x101_64x4d_fpn_20e_lvis.pth  7 \
# --out results/tao_val/gs_htc_x101_result.pkl --eval bbox
 
 
# htc_deconv_x101
#CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/tao/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_lvis.py \
#./data/pretrained_models/htc_dconv_x64_319.pth  7 \
# --out results/tao_val/htc_dconv_x101_result.pkl --eval bbox

# gs_htc_dconv_x101
# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./tools/dist_test_lvis.sh configs/tao/gs_htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_lvis.py \
#./data/pretrained_models/gs_htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_lvis.pth  7 \
# --out results/tao_val/gs_htc_dconv_x101_result.pkl --eval bbox
 
 

 