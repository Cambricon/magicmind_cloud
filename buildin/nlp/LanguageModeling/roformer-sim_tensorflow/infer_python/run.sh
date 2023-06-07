#!/bin/bash
MAGICMIND_MODEL=${1}

if [ ! $# -eq 1 ];
then
    echo "bash run.sh <magicmind_model>"
    echo "Usage: bash run.sh ${MODEL_PATH}/roformer_force_float32_false_2_8"
    exit -1
fi

cd $PROJ_ROOT_PATH/infer_python

python infer.py --magicmind_model ${MAGICMIND_MODEL} \
                --vocab_file $PROJ_ROOT_PATH/data/models/chinese_roformer-sim-char-ft_L-6_H-384_A-6/vocab.txt \
                --device_id 0