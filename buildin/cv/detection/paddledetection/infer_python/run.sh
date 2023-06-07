set -x
set -e

magicmind_model=${1}

cd ${PROJ_ROOT_PATH}/export_model/PaddleDetection
# only support batch_size=1
python tools/eval.py -c ${PADDLEDETECTION_MODEL_CONFIG_PATH} \
                     -o use_gpu=false \
                     EvalReader.batch_size=1  \
                     EvalDataset.dataset_dir=${COCO_DATASETS_PATH} \
                     weights=${magicmind_model}