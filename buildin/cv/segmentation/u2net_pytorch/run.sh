#!/bin/bash
set -e
set -x

### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
#bash run.sh parameter_id batch_size
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh parameter_id quant_mode shape_mutable batch_size
bash run.sh force_float32 1

### 2.infer and eval
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh parameter_id quant_mode shape_mutable batch_size batch
bash run.sh force_float32 1

### 3.perf
#bash $PROJ_ROOT_PATH/benchmark/perf.sh parameter_id quant_mode shape_mutable batch_size batch threads
bash $PROJ_ROOT_PATH/benchmark/perf.sh force_float32 1 1

###4. compare eval and perf result
python $MAGICMIND_CLOUD/test/compare_eval.py --metric u2net --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_1/result.txt --output_ok_file $PROJ_ROOT_PATH/data/output_ok/force_float32_1_eval --model u2net_pytorch 
python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/force_float32_1_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/force_float32_1_log_perf --model u2net_pytorch


