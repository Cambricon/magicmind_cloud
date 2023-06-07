#!/bin/bash
set -e
set -x

magicmind_model=${1}
batch_size=${2}
image_num=${3}
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi

python infer.py --magicmind_model ${magicmind_model} \
                --image_num ${image_num} \
		--data_folder ${CARVANA_DATASETS_PATH}/ \
                --batch_size ${batch_size} \
		--output_folder ${PROJ_ROOT_PATH}/data/output \

