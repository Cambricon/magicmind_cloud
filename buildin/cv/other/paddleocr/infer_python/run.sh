set -x
set -e

magicmind_det_model=${1}
magicmind_rec_model=${2}
magicmind_cls_model=${3}

# note: 1.the magicmind path is relative to the PaddleOCR code. 
cd ${PROJ_ROOT_PATH}/export_model/PaddleOCR
python tools/eval.py --config ${PADDLEOCR_DET_MODEL_CONFIG_PATH} \
                     -o Global.use_gpu=false \
                        Global.checkpoints=${magicmind_det_model} \
                        Eval.dataset.data_dir=${ICDAR2015_DATASETS_PATH}/det \
                        Eval.dataset.label_file_list=${MODEL_PATH}/test_icdar2015_label.txt

cd ${PROJ_ROOT_PATH}/export_model/PaddleOCR
python tools/eval.py --config ${PADDLEOCR_REC_MODEL_CONFIG_PATH} \
                     -o Global.use_gpu=false \
                        Global.checkpoints=${magicmind_rec_model} \
                        Eval.dataset.data_dir=${ICDAR2015_DATASETS_PATH}/rec \
                        Eval.loader.batch_size_per_card=32 \
                        Eval.dataset.label_file_list=${MODEL_PATH}/rec_gt_test.txt
                        

                