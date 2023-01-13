#!/bin/bash
PRECISION=$1 #force_float32/force_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
MAX_SEQ_LENGTH=$4

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
else 
    echo "folder: $PROJ_ROOT_PATH/data/output already exits"
fi

if [ -d "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH}" ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH}"
    echo "mkdir sucessed!!!"
fi
echo "infer Magicmind model..."
python infer.py --device_id 0 \
                --magicmind_model $MODEL_PATH/bert_tensorflow_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} \
                --squad_json $DATASETS_PATH/dev-v1.1.json \
                --vocab_file $PROJ_ROOT_PATH/data/vocab.txt \
                --batch_size ${BATCH_SIZE} \
                --max_seq_length ${MAX_SEQ_LENGTH} \
                --output_dir $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH}
python $MAGICMIND_CLOUD/buildin/thirdparty/squad_evaluate.py $DATASETS_PATH/dev-v1.1.json \
                $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH}/predictions.json \
                $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH}/result.txt
 
                               
