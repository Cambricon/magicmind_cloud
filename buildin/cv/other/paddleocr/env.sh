# 开始运行本仓库前,先检查数据集路径是否存在
# 若不存在则根据您的实际路径修改
#export ICDAR2015_DATASETS_PATH=
if [ -z ${ICDAR2015_DATASETS_PATH} ] || [ ! -d ${ICDAR2015_DATASETS_PATH} ];then
    echo "Error: ICDAR2015_DATASETS_PATH is not found, please set it and export it to env!"
fi

export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径
export PROJ_ROOT_PATH=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
export MAGICMIND_CLOUD=${PROJ_ROOT_PATH%buildin*}
export MODEL_PATH=${PROJ_ROOT_PATH}/data/models

#cv类网络通用文件路径
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/cv/utils

# Python公共组件路径
export PYTHON_COMMON_PATH=${MAGICMIND_CLOUD}/buildin/python_common

has_add_common_path=$(echo ${PYTHONPATH}|grep "${PYTHON_COMMON_PATH}")
if [ -z ${has_add_common_path} ];then
    export PYTHONPATH=${PYTHONPATH}:${PYTHON_COMMON_PATH}
fi

has_add_util_path=$(echo ${PYTHONPATH}|grep "${UTILS_PATH}")
if [ -z ${has_add_util_path} ];then
    export PYTHONPATH=${PYTHONPATH}:${UTILS_PATH}
fi

declare -A config_paths 
declare -A pretrained_paths
config_paths=( 
    [ch_PP-OCRv2_det_infer]="${PROJ_ROOT_PATH}/export_model/PaddleOCR/configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml" \
    [ch_PP-OCRv2_rec_infer]="${PROJ_ROOT_PATH}/export_model/PaddleOCR/configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml" \
    [ch_ppocr_mobile_v2.0_cls_infer]="${PROJ_ROOT_PATH}/export_model/PaddleOCR/configs/cls/cls_mv3.yml" \
    [ch_PP-OCRv3_det_infer]="${PROJ_ROOT_PATH}/export_model/PaddleOCR/configs/det/det_mv3_db.yml" \
    [ch_PP-OCRv3_rec_infer]="${PROJ_ROOT_PATH}/export_model/PaddleOCR/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml"
)
pretrained_paths=( 
    [ch_PP-OCRv2_det_infer]="https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar" \
    [ch_PP-OCRv2_rec_infer]="https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar" \
    [ch_ppocr_mobile_v2.0_cls_infer]="https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar" \
    [ch_PP-OCRv3_det_infer]="https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar" \
    [ch_PP-OCRv3_rec_infer]="https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar"
)

# 用户可自行选择其中一个模型，项目将基于用户所选模型进行MagicMind 模型构建，模型推理测试以及模型性能测试
# 指定完成之后 务必执行source env.sh 使该环境变量生效
# PaddleOCR model version: OCRv2、OCRv3
OCR_VERSION=OCRv3

if [ ${OCR_VERSION} = "OCRv2" ];then
    export PADDLEOCR_DET_MODEL_NAME=ch_PP-OCRv2_det_infer
    export PADDLEOCR_REC_MODEL_NAME=ch_PP-OCRv2_rec_infer
    export PADDLEOCR_CLS_MODEL_NAME=ch_ppocr_mobile_v2.0_cls_infer
else 
    export PADDLEOCR_DET_MODEL_NAME=ch_PP-OCRv3_det_infer 
    export PADDLEOCR_REC_MODEL_NAME=ch_PP-OCRv3_rec_infer
    export PADDLEOCR_CLS_MODEL_NAME=ch_ppocr_mobile_v2.0_cls_infer
fi

export PADDLEOCR_DET_MODEL_CONFIG_PATH=${config_paths[${PADDLEOCR_DET_MODEL_NAME}]}
export PADDLEOCR_REC_MODEL_CONFIG_PATH=${config_paths[${PADDLEOCR_REC_MODEL_NAME}]}
export PADDLEOCR_CLS_MODEL_CONFIG_PATH=${config_paths[${PADDLEOCR_CLS_MODEL_NAME}]}

export PADDLEOCR_DET_MODEL_PRETRAINED_PATH=${pretrained_paths[${PADDLEOCR_DET_MODEL_NAME}]}
export PADDLEOCR_REC_MODEL_PRETRAINED_PATH=${pretrained_paths[${PADDLEOCR_REC_MODEL_NAME}]}
export PADDLEOCR_CLS_MODEL_PRETRAINED_PATH=${pretrained_paths[${PADDLEOCR_CLS_MODEL_NAME}]}

echo "PaddleOCR Model Version is: ${OCR_VERSION}"
echo "PaddleOCR det Model is: ${PADDLEOCR_DET_MODEL_NAME}"
echo "PaddleOCR rec Model is: ${PADDLEOCR_REC_MODEL_NAME}"
echo "PaddleOCR cls Model is: ${PADDLEOCR_CLS_MODEL_NAME}"

