### 在开始运行本仓库前先检查以下路径：
echo "check below paths before run the sample!!!"
export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=$MAGICMIND_CLOUD/buildin/nlp/SpeechSynthesis/tacotron2_onnx
#数据集路径
export DATASETS_PATH=/nfsdata/modelzoo/datasets/phrases
#模型路径
export MODEL_PATH=$PROJ_ROOT_PATH/data/models/
#nlp类网络通用文件路径
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/nlp/utils
