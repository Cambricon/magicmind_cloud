# 开始运行本仓库前,先检查数据集路径是否存在
# 若不存在则根据您的实际路径修改
#export UCF101_DATASETS_PATH=
if [ -z ${UCF101_DATASETS_PATH} ] || [ ! -d ${UCF101_DATASETS_PATH} ];then
    echo "Error: UCF101_DATASETS_PATH is not found, please set it and export it to env!"
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
export CPP_COMMON_PATH=${MAGICMIND_CLOUD}/buildin/cpp_common

has_add_python_common_path=$(echo ${PYTHONPATH}|grep "${PYTHON_COMMON_PATH}")
if [ -z ${has_add_python_common_path} ];then
    export PYTHONPATH=${PYTHONPATH}:${PYTHON_COMMON_PATH}
fi

has_add_cpp_common_path=$(echo ${PYTHONPATH}|grep "${CPP_COMMON_PATH}")
if [ -z ${has_add_cpp_common_path} ];then
    export PYTHONPATH=${PYTHONPATH}:${CPP_COMMON_PATH}
fi

has_add_util_path=$(echo ${PYTHONPATH}|grep "${UTILS_PATH}")
if [ -z ${has_add_util_path} ];then
    export PYTHONPATH=${PYTHONPATH}:${UTILS_PATH}
fi
#export MLU_VISIBLE_DEVICES=7
