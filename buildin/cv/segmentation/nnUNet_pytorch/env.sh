### 在开始运行本仓库前先检查以下路径：
export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=$MAGICMIND_CLOUD/buildin/cv/segmentation/nnUNet_pytorch
#数据集路径
export DATASETS_PATH=/nfsdata/modelzoo/datasets/nnUNet_dataset
export nnUNet_raw_data_base=$DATASETS_PATH/nnUNet_raw_data_base
export nnUNet_preprocessed=$DATASETS_PATH/nnUNet_preprocessed
# nnUNet模型保存路径
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
