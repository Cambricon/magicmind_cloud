### 在开始运行本仓库前先检查以下路径：
#数据集路径
#export CITYSCAPES_DATASETS_PATH
if [ -z ${CITYSCAPES_DATASETS_PATH} ] || [ ! -d ${CITYSCAPES_DATASETS_PATH} ];then
    echo "Error: CITYSCAPES_DATASETS_PATH is not found, please set it and export it to env!"
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
    [OCRNet]="${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes.py" \
    [DeepLabV3]="${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/deeplabv3/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes.py" \
    [UNet]="${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py" \
    [DeepLabV3Plus]="${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes.py"
    [HRNet]="${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py"
    [MobileNetV2]="${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes.py" \
    [ResNest]="${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes.py" \
    [FastSCNN]="${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py" \
)

pretrained_paths=( 
    [OCRNet]="https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes/ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001-b9172d0c.pth" \
    [DeepLabV3]="https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes_20200908_005644-57bb8425.pth" \
    [UNet]="https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth" \
    [DeepLabV3Plus]="https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes_20200908_005644-ee6158e0.pth" \
    [HRNet]="https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_80k_cityscapes/fcn_hr18_512x1024_80k_cityscapes_20200601_223255-4e7b345e.pth" \
    [MobileNetV2]="https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes/fcn_m-v2-d8_512x1024_80k_cityscapes_20230227-144821-0d3a4e51.pth" \
    [ResNest]="https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes/deeplabv3_s101-d8_512x1024_80k_cityscapes_20200807_144429-b73c4270.pth" \
    [FastSCNN]="https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth" \
)

model_image_sizes=(
    [OCRNet]="1024,2048" \
    [DeepLabV3]="1024,2048" \
    [UNet]="1024,2048" \
    [DeepLabV3Plus]="1024,2048" \
    [HRNet]="1024,2048" \
    [MobileNetV2]="1024,2048" \
    [ResNest]="1024,2048" \
    [FastSCNN]="1024,2048" \
)

# 用户可自行选择其中一个模型，项目将基于用户所选模型进行MagicMind模型构建，模型推理测试以及模型性能测试
# 索引从0起始 指定完成之后 务必执行source env.sh 使该环境变量生效
# HTC暂时无效
MMSEGMENTATION_MODEL_NAMES=(
    OCRNet \
    DeepLabV3 \
    UNet \
    DeepLabV3Plus \
    HRNet \
    MobileNetV2 \
    ResNest \
    FastSCNN \
)

# choose which model to build and infer index begin from 0
export MMSEGMENTATION_MODEL_NAME=${MMSEGMENTATION_MODEL_NAMES[2]}
# choose model image size
export MMSEGMENTATION_MODEL_IMAGE_SIZE=${model_image_sizes[${MMSEGMENTATION_MODEL_NAME}]}
# choose config file path
export MMSEGMENTATION_MODEL_CONFIG_PATH=${config_paths[${MMSEGMENTATION_MODEL_NAME}]}
# choose pretrained file path
export MMSEGMENTATION_MODEL_PRETRAINED_PATH=${pretrained_paths[${MMSEGMENTATION_MODEL_NAME}]}

echo "MMSegmentation Model: ${MMSEGMENTATION_MODEL_NAME}"
echo "MMSegmentation Model Config URL: ${MMSEGMENTATION_MODEL_CONFIG_PATH}"
echo "MMSegmentation Model Pretrained URL: ${MMSEGMENTATION_MODEL_PRETRAINED_PATH}"
echo "MMSegmentation Model Image Size: ${MMSEGMENTATION_MODEL_IMAGE_SIZE}"
