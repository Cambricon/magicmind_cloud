#!/bin/bash
set -e
set -x

### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash get_datasets_and_models.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode shape_mutable batch_size
bash run.sh force_float32 true 1

### 2.1 infer_python
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh quant_mode shape_mutable batch_size batch
bash run.sh force_float32 true 1 1

### 2.2 infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh

### 3.eval and perf
#bash $PROJ_ROOT_PATH/benchmark/eval.sh quant_mode shape_mutable batch ways
bash $PROJ_ROOT_PATH/benchmark/eval.sh force_float32 true 1 infer_python
#bash $PROJ_ROOT_PATH/benchmark/perf.sh quant_mode shape_mutable batch_size batch threads
bash $PROJ_ROOT_PATH/benchmark/perf.sh force_float32 true 1 1 1

###4. compare eval and perf result
python $MAGICMIND_CLOUD/test/compare_eval.py --metric vocmAP --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/voc_preds/log_eval --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_python_output_force_float32_true_1_log_eval --model ssd_caffe
python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/force_float32_true_1_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/force_float32_true_1_log_perf --model ssd_caffe

