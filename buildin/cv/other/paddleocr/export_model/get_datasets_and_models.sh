#!/bin/bash
set -e
set -x

function get_model(){
	model_addr="${1}"
	local_model_path="${2}"
	model_name=$(basename "${local_model_path}")

    if [ ! -d "${local_model_path}" ] || [ "`ls -A ${local_model_path}`" = "" ];then
        echo "Downloading "${model_name}" paddle model"
        wget -c "${model_addr}" 
        tar -xvf "${model_name}.tar" 
    else
        echo " ${model_name} paddle model exists, no need to download."
    fi
}

det_infer_model_path="${MODEL_PATH}/${PADDLEOCR_DET_MODEL_NAME}"
rec_infer_model_path="${MODEL_PATH}/${PADDLEOCR_REC_MODEL_NAME}"
cls_infer_model_path="${MODEL_PATH}/${PADDLEOCR_CLS_MODEL_NAME}"

if [ ! -d ${MODEL_PATH} ];then
    mkdir -p ${MODEL_PATH}
fi 

# get the model you nedd
pushd ${MODEL_PATH}
    get_model "${PADDLEOCR_DET_MODEL_PRETRAINED_PATH}" "${det_infer_model_path}"
    get_model "${PADDLEOCR_REC_MODEL_PRETRAINED_PATH}" "${rec_infer_model_path}"
    get_model "${PADDLEOCR_CLS_MODEL_PRETRAINED_PATH}" "${cls_infer_model_path}"
popd

# start to preprae dataset
# data path dir:
# ICDAR2015/
# ├── det
# │   └── ch4_test_images  # det image path
# └── rec
#     └── test  # rec image path

if [ ! -d ${ICDAR2015_DATASETS_PATH} ];then
    mkdir -p ${ICDAR2015_DATASETS_PATH}
    echo "Please download ICDAR 2015 det and rec test data from https://rrc.cvc.uab.es/?ch=4&com=downloads and unzip it to ${ICDAR2015_DATASETS_PATH}"
    exit 1
fi 

# prepare label data
test_labe_path="${MODEL_PATH}/test_icdar2015_label.txt"
if [ ! -f "${test_labe_path}" ];then
    echo "Downloading test_icdar2015_label.txt"
    wget -P ${MODEL_PATH} -c https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt
    wget -P ${MODEL_PATH} -c https://paddleocr.bj.bcebos.com/dataset/rec_gt_test.txt
fi

# clone PaddleOCR
cd ${PROJ_ROOT_PATH}/export_model 
PaddleOCR=PaddleOCR
if [ ! -d ${PaddleOCR} ]; then
    git clone https://github.com/PaddlePaddle/PaddleOCR.git
    cd ${PaddleOCR}
    git checkout 4b8e333f102bcee9b1a0685a3f709b72ca800810       
fi

# patch
cd ${PROJ_ROOT_PATH}/export_model 
if grep -q "MMRunner" ${PROJ_ROOT_PATH}/export_model/PaddleOCR/tools/eval.py;
then
    echo "mm_backend.patch already be used"
else
    git apply mm_backend.patch
fi