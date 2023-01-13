#!/bin/bash
PRECISION=$1
BATCH_SIZE_MIN=$2
BATCH_SIZE=$3
BATCH_SIZE_MAX=$4
INPUT_LEN=$5
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
else
    echo "folder: $PROJ_ROOT_PATH/data/output already exits"
fi
echo "infer Magicmind model..."
python infer.py  --models_dir $MODEL_PATH \
                 -o $PROJ_ROOT_PATH/data/output \
                 -warmup_iters 3 \
                 --num_iters 100 \
                 -il $INPUT_LEN \
                 -bs $BATCH_SIZE_MIN,$BATCH_SIZE,$BATCH_SIZE_MAX \
                 --log_file $PROJ_ROOT_PATH/data/output/${PRECISION}_${BATCH_SIZE_MIN}_${BATCH_SIZE}_${BATCH_SIZE_MAX}_${INPUT_LEN}_log_perf \
                 --precision $PRECISION \
                 --device 0 \
                 --no-waveglow
