#!/bin/bash
QUANT_MODE=$1 #force_float32/force_float16
BATCH=$2
INPUT_LEN=$3
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
else
    echo "folder: $PROJ_ROOT_PATH/data/output already exits"
fi

echo "infer Magicmind model..."
python infer.py  --models_dir ${PROJ_ROOT_PATH}/data/models \
                 -o ${PROJ_ROOT_PATH}/data/output \
                 -warmup_iters 3 \
                 --num_iters 100 \
                 -il ${INPUT_LEN} \
                 -bs ${BATCH} \
                 --log_file ${PROJ_ROOT_PATH}/data/output/${QUANT_MODE}_${BATCH}_${INPUT_LEN}_log_perf \
                 --quant_mode ${QUANT_MODE} \
                 --device 0 \
                 --no-waveglow
