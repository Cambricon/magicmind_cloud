### 在开始运行本仓库前先检查以下路径：
echo "check below paths before run the sample!!!"
export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=${NEUWARE_HOME}/bin
#本sample工作路径
export PROJ_ROOT_PATH=${PWD}
#数据集路径
#export CITYSCAPES_DATASETS_PATH=/path/to/modelzoo/datasets/cityscapes
#模型路径
export MODEL_PATH=${PROJ_ROOT_PATH}/data/models

echo "check CITYSCAPES_DATASETS_PATH before run the sample!!!"
echo "CITYSCAPES_DATASETS_PATH now is $CITYSCAPES_DATASETS_PATH, please replace it to path where you want to save datasets"
echo "NEUWARE_HOME now is $NEUWARE_HOME"
echo "MM_RUN_PATH now is $MM_RUN_PATH"
echo "PROJ_ROOT_PATH is $PROJ_ROOT_PATH"
echo "MODEL_PATH is $MODEL_PATH"
