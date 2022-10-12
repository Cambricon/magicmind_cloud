#!/bin/bash
set -e
set -x

###0. convert model
cd $PROJ_ROOT_PATH/export_model
#bash run.sh batch_size max_seq_len
bash run.sh

###1. gen_model - qint8_mixed_float16  force_float16 force_float32
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode shape_mutable batch_size max_seq_len
bash run.sh force_float16 true 1 384 

###2. infer_python and eval
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh quant_mode shape_mutable batch_size batch max_seq_len
bash run.sh force_float16 true 1 384

###3. perf
#bash $PROJ_ROOT_PATH/benchmark/perf.sh quant_mode shape_mutable batch_size max_seq_len threads
bash $PROJ_ROOT_PATH/benchmark/perf.sh force_float16 true 1 384 1

###4. compare eval and perf result
python $MAGICMIND_CLOUD/test/compare_eval.py --metric squad --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float16_true_1_384/result.txt --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_python_output_force_float16_true_1_384_result.txt --model bert_tensorflow
python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/force_float16_true_1_384_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/force_float16_true_1_384_log_perf --model bert_tensorflow

