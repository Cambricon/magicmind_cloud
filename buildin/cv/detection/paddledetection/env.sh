# 开始运行本仓库前,先检查数据集路径是否存在
# 若不存在则根据您的实际路径修改
#export COCO_DATASETS_PATH=/path/to/modelzoo/datasets/coco/
if [ -z ${COCO_DATASETS_PATH} ] || [ ! -d ${COCO_DATASETS_PATH} ];then
    echo "Error: COCO_DATASETS_PATH is not found, please set it and export it to env!"
fi

export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=${NEUWARE_HOME}/bin
#本sample工作路径
export PROJ_ROOT_PATH=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
export MAGICMIND_CLOUD=${PROJ_ROOT_PATH%buildin*}
export MODEL_PATH=${PROJ_ROOT_PATH}/data/models

# CV类网络通用文件路径
export UTILS_PATH=${MAGICMIND_CLOUD}/buildin/cv/utils

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
declare -A model_sizes
config_paths=( 
    [yolov3_darknet53_270e_coco]="${PROJ_ROOT_PATH}/export_model/PaddleDetection/configs/yolov3/yolov3_darknet53_270e_coco.yml" \
    [ppyoloe_crn_s_400e_coco]="${PROJ_ROOT_PATH}/export_model/PaddleDetection/configs/ppyoloe/ppyoloe_crn_s_400e_coco.yml" 
)
pretrained_paths=( 
    [yolov3_darknet53_270e_coco]="https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams" \
    [ppyoloe_crn_s_400e_coco]="https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams" 
)
model_sizes=( 
    [yolov3_darknet53_270e_coco]=608 \
    [ppyoloe_crn_s_400e_coco]=640 
)

# 用户可自行选择其中一个模型，项目将基于用户所选模型进行MagicMind 模型构建，模型推理测试以及模型性能测试
# 指定完成之后 务必执行source env.sh 使该环境变量生效
# YOLOv3 PPYOLOE
MODEL_NAME=PPYOLOE

if [ ${MODEL_NAME} = "YOLOv3" ];then
    export PADDLEDETECTION_MODEL_NAME=yolov3_darknet53_270e_coco
elif [ ${MODEL_NAME} = "PPYOLOE" ];then
    export PADDLEDETECTION_MODEL_NAME=ppyoloe_crn_s_400e_coco
fi

export PADDLEDETECTION_MODEL_CONFIG_PATH=${config_paths[${PADDLEDETECTION_MODEL_NAME}]}

export PADDLEDETECTION_MODEL_PRETRAINED_PATH=${pretrained_paths[${PADDLEDETECTION_MODEL_NAME}]}

export PADDLEDETECTION_MODEL_INPUT_SIZE=${model_sizes[${PADDLEDETECTION_MODEL_NAME}]}

echo "PADDLEDETECTION Model Name is: ${OCR_VERSION}"
echo "PADDLEDETECTION Model is: ${PADDLEDETECTION_MODEL_NAME}"
echo "PADDLEDETECTION Config Path is: ${PADDLEDETECTION_MODEL_CONFIG_PATH}"
echo "PADDLEDETECTION Model Input Size is: ${PADDLEDETECTION_MODEL_INPUT_SIZE}"
echo "PADDLEDETECTION Pretrained Path is: ${PADDLEDETECTION_MODEL_PRETRAINED_PATH}"
