#!/bin/bash
set -e
set -x

# 0. convert model
pip install -r requirement.txt
cd $PROJ_ROOT_PATH/export_model 
bash run.sh 1

# 1. gen_model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode shape_mutable batch_size conf iou max_det
bash run.sh force_float32 true 1 0.001 0.65 1000

# 2.1 build infer_cpp and infer
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh quant_mode shape_mutable batch_size image_num
bash run.sh force_float32 true 1 1000
# 2.2 infer_python
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh quant_mode shape_mutable batch_size batch image_num
bash run.sh force_float32 true 1 1 1000

## 3.eval and perf
#bash $PROJ_ROOT_PATH/benchmark/eval.sh quant_mode shape_mutable batch_size image_num infer_python/infer_cpp
bash $PROJ_ROOT_PATH/benchmark/eval.sh force_float32 true 1 1000 infer_python
bash $PROJ_ROOT_PATH/benchmark/eval.sh force_float32 true 1 1000 infer_cpp
#bash $PROJ_ROOT_PATH/benchmark/perf.sh quant_mode shape_mutable batch_size batch threads
bash $PROJ_ROOT_PATH/benchmark/perf.sh force_float32 true 1 1 1

###4. compare eval and perf result
python $MAGICMIND_CLOUD/test/compare_eval.py --metric cocomAP --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/log_eval --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_python_output_force_float32_true_1_log_eval --model yolov5_v6_1_pytorch
python $MAGICMIND_CLOUD/test/compare_eval.py --metric cocomAP --output_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/log_eval --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_python_output_force_float32_true_1_log_eval --model yolov5_v6_1_pytorch
python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/force_float32_true_1_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/force_float32_true_1_log_perf --model yolov5_v6_1_pytorch 
