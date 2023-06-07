#!/bin/bash
set -e
set -x


###1. convert model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###2. gen_model -  force_float16 force_float32
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape> <max_seq_length>
bash run.sh ${MODEL_PATH}/roformer_force_float32_true_2_8 force_float32 2 true 8

###3. infer_python 
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <magicmind_model>
bash run.sh ${MODEL_PATH}/roformer_force_float32_true_2_8
