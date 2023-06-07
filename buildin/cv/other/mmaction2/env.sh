### 在开始运行本仓库前先检查以下路径：
#数据集路径
#export KINETICS_POSTPROCESS_DATASETS_PATH=
if [ -z ${KINETICS_POSTPROCESS_DATASETS_PATH} ] || [ ! -d ${KINETICS_POSTPROCESS_DATASETS_PATH} ];then
    echo "Error: KINETICS_POSTPROCESS_DATASETS_PATH is not found, please set it and export it to env!"
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
    [I3D]="${PROJ_ROOT_PATH}/export_model/mmaction2/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py" \
    [TSM]="${PROJ_ROOT_PATH}/export_model/mmaction2/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py" \
)

pretrained_paths=( 
    [I3D]="https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth" \
    [TSM]="https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth" \
)

model_image_sizes=(
    [I3D]="30 3 32 256 256" \
    [TSM]="8 3 224 224" \
)

# 用户可自行选择其中一个模型，项目将基于用户所选模型进行MagicMind模型构建，模型推理测试以及模型性能测试
# 索引从0起始 指定完成之后 务必执行source env.sh 使该环境变量生效
MMACTION2_MODEL_NAMES=(
    I3D \
    TSM \
)

# choose which model to build and infer index begin from 0
export MMACTION2_MODEL_NAME=${MMACTION2_MODEL_NAMES[1]}
# choose model image size
export MMACTION2_MODEL_IMAGE_SIZE=${model_image_sizes[${MMACTION2_MODEL_NAME}]}
# choose config file path
export MMACTION2_MODEL_CONFIG_PATH=${config_paths[${MMACTION2_MODEL_NAME}]}
# choose pretrained file path
export MMACTION2_MODEL_PRETRAINED_PATH=${pretrained_paths[${MMACTION2_MODEL_NAME}]}

echo "MMAction2 Model: ${MMACTION2_MODEL_NAME}"
echo "MMAction2 Model Config URL: ${MMACTION2_MODEL_CONFIG_PATH}"
echo "MMAction2 Model Pretrained URL: ${MMACTION2_MODEL_PRETRAINED_PATH}"
echo "MMAction2 Model Image Size: ${MMACTION2_MODEL_IMAGE_SIZE}"
