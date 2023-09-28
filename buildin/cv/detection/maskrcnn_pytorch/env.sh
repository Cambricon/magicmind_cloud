### 在开始运行本仓库前先检查以下路径：
export NEUWARE_HOME=/usr/local/neuware/
##MM_RUN路径
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径  
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=$MAGICMIND_CLOUD/buildin/cv/detection/maskrcnn_pytorch
#数据集路径
#export COCO_DATASETS_PATH=/path/to/modelzoo/datasets/coco
#cv类网络通用文件路径
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/cv/utils
#模型路径
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
echo "check COCO_DATASETS_PATH before run the sample!!!"
echo "COCO_DATASETS_PATH now is $COCO_DATASETS_PATH, please replace it to the path you want to save the datasets"
echo "NEUWARE_HOME now is $NEUWARE_HOME"
echo "MM_RUN_PATH now is $MM_RUN_PATH"
echo "MAGICMIND_CLOUD is $MAGICMIND_CLOUD"
echo "PROJ_ROOT_PATH is $PROJ_ROOT_PATH"
echo "MODEL_PATH is $MODEL_PATH"
echo "UTILS_PATH is $UTILS_PATH"
