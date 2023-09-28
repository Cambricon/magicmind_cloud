### 在开始运行本仓库前先检查以下路径：
export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径
export PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
export MAGICMIND_CLOUD="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
#数据集路径
#export MSRA_B_DATASETS_PATH=/path/to/modelzoo/datasets/MSRA-B
# 模型保存路径
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/cv/utils 

echo "check below paths before run this sample!!!"
echo "MSRA_B_DATASETS_PATH now is $MSRA_B_DATASETS_PATH, please replace it to path where you want to save datasets"
echo "NEUWARE_HOME now is $NEUWARE_HOME"
echo "MM_RUN_PATH now is $MM_RUN_PATH"
echo "MAGICMIND_CLOUD is $MAGICMIND_CLOUD"
echo "PROJ_ROOT_PATH is $PROJ_ROOT_PATH"
echo "MODEL_PATH is $MODEL_PATH"
echo "UTILS_PATH is $UTILS_PATH"
