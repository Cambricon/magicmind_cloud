#!/bin/bash
set -e
set -x
MAGICMIND_MODEL=${1}
BATCH_SIZE=${2:-1} #force_float32/force_float16
MAX_SEQ_LENGTH=${3:-384}

if [ $# -lt 1 -o $# -gt 3 ];
then
    echo "bash run.sh <magicmind_model> <batch_size> <max_seq_length> "
    echo "Usage: bash run.sh ${MODEL_PATH}/roformer_force_float32_false_1_384 1 384"
    exit -1
fi

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
else 
    echo "folder: $PROJ_ROOT_PATH/data/output already exits"
fi

output_dir=${PROJ_ROOT_PATH}/data/output/$(basename ${MAGICMIND_MODEL})
echo ${output_dir}
if [ -d ${output_dir} ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir ${output_dir}
    echo "mkdir sucessed!!!"
fi
echo "infer Magicmind model..."
python infer.py --device_id 0 \
                --magicmind_model ${MAGICMIND_MODEL} \
                --squad_json ${SQUAD_DATASETS_PATH}/dev-v1.1.json \
                --vocab_file ${PROJ_ROOT_PATH}/data/vocab.txt \
                --batch_size ${BATCH_SIZE} \
                --max_seq_length ${MAX_SEQ_LENGTH} \
                --output_dir ${output_dir}

python $MAGICMIND_CLOUD/buildin/thirdparty/squad_evaluate.py \
                ${SQUAD_DATASETS_PATH}/dev-v1.1.json \
                ${output_dir}/predictions.json \
                ${output_dir}/result.txt
 
                               
