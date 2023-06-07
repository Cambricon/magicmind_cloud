set -e
set -x

magicmind_model=${1}
batch_size=${2}
output_pkl="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model}).pkl"

# 调用MMDetection接口
cd $PROJ_ROOT_PATH/export_model/mmsegmentation/tools/

splits=(${MMSEGMENTATION_MODEL_PRETRAINED_PATH//// })
pth_file=${splits[-1]}
python test.py ${MMSEGMENTATION_MODEL_CONFIG_PATH} \
               ${magicmind_model} \
               --out ${output_pkl} \
               --eval mIoU \
               --backend magicmind \
               --device_id 0 \
               --batch_size ${batch_size} \