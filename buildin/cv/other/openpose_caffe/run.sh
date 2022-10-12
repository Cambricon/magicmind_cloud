#!/bin/bash
set -e
set -x

# 0. convert model
cd $PROJ_ROOT_PATH/export_model 
bash run.sh

# 1. gen_model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode batch_size
bash run.sh qint8_mixed_float16 1

# 2 build infer_cpp and infer
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh quant_mode batch_size
bash run.sh qint8_mixed_float16 1

### 3.eval and perf
#bash $PROJ_ROOT_PATH/benchmark/eval.sh quant_mode batch_size
bash $PROJ_ROOT_PATH/benchmark/eval.sh qint8_mixed_float16 1
#bash $PROJ_ROOT_PATH/benchmark/perf.sh quant_mode batch_size threads
bash $PROJ_ROOT_PATH/benchmark/perf.sh qint8_mixed_float16 1 1

###4. compare eval and perf result
python $MAGICMIND_CLOUD/test/compare_eval.py --metric cocoKeyPoints --output_file $PROJ_ROOT_PATH/data/images/qint8_mixed_float16_1_eval --output_ok_file $PROJ_ROOT_PATH/data/images/qint8_mixed_float16_1_eval --model openpose_caffe
python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/images/pose_body25_qint8_mixed_float16_1_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/pose_body25_qint8_mixed_float16_1_log_perf --model pose_body25
python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/images/pose_coco_qint8_mixed_float16_1_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/pose_coco_qint8_mixed_float16_1_log_perf --model pose_coco
