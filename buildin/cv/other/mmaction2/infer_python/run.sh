set -e
set -x

magicmind_model=${1}
batch_size=${2}
output_json="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model}).json"

# 调用MMDetection接口
cd $PROJ_ROOT_PATH/export_model/mmaction2/tools/
python test.py ${MMACTION2_MODEL_CONFIG_PATH} \
               ${magicmind_model} \
               --out ${output_json} \
               --eval top_k_accuracy \
               --average-clips prob \
               --magicmind \
               --device_id 0 \
               --batch_size ${batch_size}