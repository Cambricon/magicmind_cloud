#!/bin/bash
set -e
set -x

###0. convert model
cd $PROJ_ROOT_PATH/export_model
#bash run.sh
bash run.sh

###1. gen_model - qint8_mixed_float16  force_float16 force_float32
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode batch_size_min batch_size batch_size_max
bash run.sh force_float16 1 4 8

###2. infer_python and eval
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh quant_mode batch input_len
bash run.sh force_float16 1 128

###4. compare eval and perf result
python $MAGICMIND_CLOUD/test/compare_perf.py --metric tacotron --output_file $PROJ_ROOT_PATH/data/output/force_float16_1_128_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/force_float16_1_128_log_perf --model tacotron_onnx

