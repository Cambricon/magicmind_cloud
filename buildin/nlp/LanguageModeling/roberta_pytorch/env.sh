### 在开始运行本仓库前先检查以下路径：
echo "check below paths before run the sample!!!"
export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=$MAGICMIND_CLOUD/buildin/nlp/LanguageModeling/roberta_pytorch
#数据集路径
export DATASETS_PATH=/nfsdata/modelzoo/datasets/chnsenticorp
#模型路径
export MODEL_PATH=$PROJ_ROOT_PATH/data/models/
#nlp类网络通用文件路径
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/nlp/utils

echo "check DATASETS_PATH before run the sample!!!"
echo "DATASETS_PATH now is $DATASETS_PATH, please replace it to the path you want to save the datasets"
echo "NEUWARE_HOME now is $NEUWARE_HOME"
echo "MM_RUN_PATH now is $MM_RUN_PATH"
echo "MAGICMIND_CLOUD is $MAGICMIND_CLOUD"
echo "PROJ_ROOT_PATH is $PROJ_ROOT_PATH"
echo "MODEL_PATH is $MODEL_PATH"
echo "UTILS_PATH is $UTILS_PATH"
