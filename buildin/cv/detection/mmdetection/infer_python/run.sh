set -e
set -x

magicmind_model=${1}
batch_size=${2}
img_num=${3}
output_pkl="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model}).pkl"

# 调用MMDetection接口
cd $PROJ_ROOT_PATH/export_model/mmdetection/tools/deployment
python test.py ${MMDETECTION_MODEL_CONFIG_PATH} \
               ${magicmind_model} \
               --out ${output_pkl} \
               --eval bbox \
               --backend magicmind \
               --device_id 0 \
               --batch_size ${batch_size} \
               --img_num ${img_num}
