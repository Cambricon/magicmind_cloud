#!/bin/bash
QUANT_MODE=$1 #force_float32/force_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
MAX_SEQ_LENGTH=$4

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
else 
    echo "folder: $PROJ_ROOT_PATH/data/output already exits"
fi

if [ -d "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}" ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}"
    echo "mkdir sucessed!!!"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${QUANT_MODE}_${SHAPE_MUTABLE}
fi

echo "infer Magicmind model..."
python infer.py --device_id 0 \
                --magicmind_model $MAGICMIND_MODEL \
                --json_file $DATASETS_PATH/dev-v1.1.json \
                --batch_size ${BATCH_SIZE} \
                --max_seq_length ${MAX_SEQ_LENGTH} \
                --compute_accuracy true \
                --output_dir $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH} \
                --acc_result $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}/acc_result.txt
