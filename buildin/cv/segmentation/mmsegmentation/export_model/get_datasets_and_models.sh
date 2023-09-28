#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"

if [ ! -d ${CITYSCAPES_DATASETS_PATH} ];then
    echo ${CITYSCAPES_DATASETS_PATH}" not exist, please download cityscapes datasets. https://www.cityscapes-dataset.com/downloads/"    
    exit 1
fi

if [ ! -d ${MODEL_PATH} ];then
    mkdir -p ${MODEL_PATH}
fi
cd ${MODEL_PATH}

splits=(${MMSEGMENTATION_MODEL_PRETRAINED_PATH//// })
local_model_save_dir=${splits[-1]}
if [ ! -f ${MODEL_PATH}/${local_model_save_dir} ];then
  wget -c ${MMSEGMENTATION_MODEL_PRETRAINED_PATH} --no-check-certificate -O ${local_model_save_dir}
fi 

cd ${PROJ_ROOT_PATH}/export_model
if [ ! -d "mmsegmentation" ];then
    git clone https://github.com/open-mmlab/mmsegmentation.git
    cd mmsegmentation
    # 切换分支
    git checkout -b v0.30.0 v0.30.0
else
    echo "mmsegmentation exist!"
fi

cd $PROJ_ROOT_PATH/export_model/mmsegmentation
if grep -q "CITYSCAPES_DATASETS_PATH" ${PROJ_ROOT_PATH}/export_model/mmsegmentation/configs/_base_/datasets/cityscapes.py;then
    echo "patch has been patched.";
else  
    git apply ../magicmind.patch
    # 安装依赖
    pip install mmcv-full==1.7.1 cityscapesscripts
    # 安装mmseg
    pip install -v -e .
fi

