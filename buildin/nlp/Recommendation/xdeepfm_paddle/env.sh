# 开始运行本仓库前,先检查数据集路径是否存在
# 若不存在则根据您的实际路径修改
# export CRITEO_DATASETS_PATH=''
if [ -z ${CRITEO_DATASETS_PATH} ] || [ ! -d ${CRITEO_DATASETS_PATH} ];then
    echo "Error: CRITEO_DATASETS_PATH is not found, please set it and export it to env!"
fi
export NEUWARE_HOME=/usr/local/neuware/
##MM_RUN路径
export MM_RUN_PATH=${NEUWARE_HOME}/bin
#本sample工作路径  
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=${MAGICMIND_CLOUD}/buildin/nlp/Recommendation/xdeepfm_paddle
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
#模型路径
export MODEL_PATH=${PROJ_ROOT_PATH}/data/models
export NETWORK=xdeepfm_paddle
