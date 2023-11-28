### 在开始运行本仓库前先检查以下路径：
export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=$NEUWARE_HOME/bin
#本sample工作路径
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=$MAGICMIND_CLOUD/buildin/cv/segmentation/nnUNet_pytorch
#数据集路径
#export NNUNET_DATASETS_PATH=/path/to/modelzoo/datasets/nnUNet_dataset
export NNUNET_nnUNet_raw_data_base=$NNUNET_DATASETS_PATH/nnUNet_raw_data_base
export DATASETS_PATH=$PROJ_ROOT_PATH/data/
export nnUNet_raw_data_base=$DATASETS_PATH/nnUNet_raw_data_base
export nnUNet_preprocessed=$DATASETS_PATH/nnUNet_preprocessed
# nnUNet模型保存路径
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
echo "check below paths before run this sample!!!"
echo "NEUWARE_HOME now is $NEUWARE_HOME"
echo "MM_RUN_PATH now is $MM_RUN_PATH"
echo "MAGICMIND_CLOUD is $MAGICMIND_CLOUD"
echo "PROJ_ROOT_PATH is $PROJ_ROOT_PATH"
echo "NNUNET_DATASETS_PATH now is $NNUNET_DATASETS_PATH, please replace it to path where you want to save datasets"
echo "NNUNET_nnUNet_raw_data_base is $NNUNET_nnUNet_raw_data_base"
echo "DATASETS_PATH now is $DATASETS_PATH, please replace it to path where you want to save datasets"
echo "nnUNet_raw_data_base is $nnUNet_raw_data_base"
echo "nnUNet_preprocessed is $nnUNet_preprocessed"
echo "MODEL_PATH is $MODEL_PATH"
