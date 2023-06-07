### 在开始运行本仓库前先检查以下路径：
#数据集路径
#export ILSVRC2012_DATASETS_PATH=
if [ -z ${ILSVRC2012_DATASETS_PATH} ] || [ ! -d ${ILSVRC2012_DATASETS_PATH} ];then
    echo "Error: ILSVRC2012_DATASETS_PATH is not found, please set it and export it to env!"
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
