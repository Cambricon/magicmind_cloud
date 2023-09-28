### 在开始运行本仓库前先检查以下路径：
echo "check below paths before run the sample!!!"
export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=$MAGICMIND_CLOUD/buildin/cv/other/fsanet_tensorflow
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
#export ALFW2000_AFLW2000_DATASETS_PATH
#cv类网络通用文件路径
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/cv/utils

echo "AFLW2000_DATASETS_PATH now is $AFLW2000_DATASETS_PATH, please replace it to path where you want to save datasets"
echo "NEUWARE_HOME now is $NEUWARE_HOME"
echo "MM_RUN_PATH now is $MM_RUN_PATH"
echo "MAGICMIND_CLOUD is $MAGICMIND_CLOUD"
echo "PROJ_ROOT_PATH is $PROJ_ROOT_PATH"
echo "MODEL_PATH is $MODEL_PATH"
echo "AFLW2000_DATASETS_PATH is $AFLW2000_DATASETS_PATH"
echo "UTILS_PATH is $UTILS_PATH"
