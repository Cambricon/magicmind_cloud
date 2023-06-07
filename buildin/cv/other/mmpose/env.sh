### 在开始运行本仓库前先检查以下路径：
#数据集路径
#export COCO_DATASETS_PATH=
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
# CPP公共接口路径
export CPP_COMMON_PATH=$MAGICMIND_CLOUD/buildin/cpp_common

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
declare -A model_image_sizes

config_paths=( 
    [HRNet]="${PROJ_ROOT_PATH}/export_model/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py" \
    [ResNet50]="${PROJ_ROOT_PATH}/export_model/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_512x512.py" \
    [MobileNetv2]="${PROJ_ROOT_PATH}/export_model/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/mobilenetv2_coco_512x512.py" \
)

pretrained_paths=( 
    [HRNet]="https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth" \
    [ResNet50]="https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512-5521bead_20200816.pth" \
    [MobileNetv2]="https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth" \
)   

model_image_sizes=(
    [HRNet]="512,512" \
    [ResNet50]="512,512" \
    [MobileNetv2]="512,512" \
)

# 用户可自行选择其中一个模型，项目将基于用户所选模型进行MagicMind模型构建，模型推理测试以及模型性能测试
# 索引从0起始 指定完成之后 务必执行source env.sh 使该环境变量生效
MMPOSE_MODEL_NAMES=(
    HRNet \
    ResNet50 \
    MobileNetv2 \
)

# choose which model to build and infer index begin from 0
export MMPOSE_MODEL_NAME=${MMPOSE_MODEL_NAMES[1]}
# choose model image size
export MMPOSE_MODEL_IMAGE_SIZE=${model_image_sizes[${MMPOSE_MODEL_NAME}]}
# choose config file path
export MMPOSE_MODEL_CONFIG_PATH=${config_paths[${MMPOSE_MODEL_NAME}]}
# choose pretrained file path
export MMPOSE_MODEL_PRETRAINED_PATH=${pretrained_paths[${MMPOSE_MODEL_NAME}]}

echo "MMPose Model: ${MMPOSE_MODEL_NAME}"
echo "MMPose Model Config URL: ${MMPOSE_MODEL_CONFIG_PATH}"
echo "MMPose Model Pretrained URL: ${MMPOSE_MODEL_PRETRAINED_PATH}"
echo "MMPose Model Image Size: ${MMPOSE_MODEL_IMAGE_SIZE}"
