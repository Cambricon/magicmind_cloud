set -e
set -x

magicmind_model=${1}
batch_size=${2}
img_num=${3}
output_pkl="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model}).pkl"

# 调用MMDetection接口
cd $PROJ_ROOT_PATH/export_model/mmpose/tools/

splits=(${MMPOSE_MODEL_PRETRAINED_PATH//// })
pth_file=${splits[-1]}
python test.py ${MMPOSE_MODEL_CONFIG_PATH} \
               ${magicmind_model} \
               --out ${output_pkl} \
               --eval mAP \
               --backend magicmind \
               --device_id 0 \
               --batch_size ${batch_size} \
               --img_num ${img_num}