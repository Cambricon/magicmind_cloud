#在开始运行本仓库前先检查以下路径：
echo "check below paths before run the sample!!!"
export NEUWARE_HOME=/usr/local/neuware/
##MM_RUN路径
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径 
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=$MAGICMIND_CLOUD/buildin/nlp/LanguageModeling/transformers_pytorch
#数据集路径 
#export GLUE_GLUE_DATASETS_PATH=
#cv类网络通用文件路径
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/nlp/utils
#模型路径
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
