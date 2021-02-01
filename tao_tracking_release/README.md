# Tracking Code

## 1. Environment setup
This code has been tested on Python 3.7.6, Pytorch 1.5.1, CUDA 10.1, please install related libraries before running this code:
```bash
pip install git+https://github.com/TAO-Dataset/tao
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install onnx==1.7.0 onnxruntime-gpu==1.4.0 opencv-python==4.0.1.24 scikit-learn==0.22.1
```
Run the following code for multiple GPU inference:
```bash
export PYTHONOPTIMIZE=1
```

## 2. Tracking

### Generate detection files from the output of the detection model:
```python
python gen_detections.py \
    --detections-dir /path/to/detection/ \
    --annotations /path/to/train.json \
    --output-dir ./data/ \
    --output-file det.txt \
    --nms-thresh 0.5 \
    --det-num 300 \
    --workers 8
```
### Extract features using the ReID model:
```python
python generate_detection_features.py \
    --detections-dir ./data/ \
    --annotations /path/to/train.json \
    --output-dir ./resources/det_onnx \
    --image-dir /path/to/frames/ \
    --detection-file det.txt \
    --model-file /path/to/reid_model.onnx \
    --gpus 0 1 2 3 4 5 6 7
```
Two ReID models can be found in `reid_pytorch/reid1.onnx` and `reid_pytorch/reid2.onnx`.
### Track with modified deepsort:
```python
python track_tao.py \
    --detections-dir ./resources/det_onnx/ \
    --annotations /path/to/train.json \
    --output-dir .results/det_onnx/ \
    --save-feature \
    --workers 8
```
### Post-processing:
`track_concat.py`: Fuse the tracking results from two detections.

`associate_track.py`: Post-associating tracklets.

`create_json_for_eval.py`: Generate json files for TAO evaluation.

`onekey_processing.py`: One file to finish the above three operations.

```python
python tao_post_processing/onekey_processing.py \
    --onnx_results1 ./results/det1_reid/ \
    --onnx_results2 ./results/det2_reid/ \
    --annotations /path/to/train.json \
    --output-dir ./results/onekey_results/ \
    --workers 8
```