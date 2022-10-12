### 在开始运行本仓库前先检查以下路径：
export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径
export PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
export MAGICMIND_CLOUD="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
#数据集路径
export DATASETS_PATH=/nfsdata/modelzoo/datasets/MSRA-B
# 模型保存路径
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/cv/utils 
