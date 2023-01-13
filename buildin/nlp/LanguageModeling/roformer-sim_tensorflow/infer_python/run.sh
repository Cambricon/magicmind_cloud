#!/bin/bash
PRECISION=$1 #force_float32/force_float16

cd $PROJ_ROOT_PATH/infer_python

python infer.py --magicmind_model $PROJ_ROOT_PATH/data/models/roformer-sim_tf_${PRECISION}_true_2_64 \
                --vocab_file $PROJ_ROOT_PATH/data/models/chinese_roformer-sim-char-ft_L-6_H-384_A-6/vocab.txt \
                --device_id 0