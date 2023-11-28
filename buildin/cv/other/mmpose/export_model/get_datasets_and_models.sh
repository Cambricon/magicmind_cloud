#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"

# 1. download datasets
if [ ! -d $COCO_DATASETS_PATH ];then
  mkdir -p $COCO_DATASETS_PATH
fi

cd $COCO_DATASETS_PATH
if [ ! -d $FILE1 ];then 
  echo "Downloading val2017.zip"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -o val2017.zip
fi

if [ ! -d $FILE2 ];then
  echo "Downloading annotations_trainval2017.zip"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -o annotations_trainval2017.zip
fi

if [ ! -f $FILE3 ];then
  echo "coco.names"
  wget -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names --no-check-certificate -O coco.names
fi


if [ ! -d ${MODEL_PATH} ];then
    mkdir -p ${MODEL_PATH}
fi
cd ${MODEL_PATH}

splits=(${MMPOSE_MODEL_PRETRAINED_PATH//// })
local_model_save_dir=${splits[-1]}
if [ ! -f ${MODEL_PATH}/${local_model_save_dir} ];then
  wget -c ${MMPOSE_MODEL_PRETRAINED_PATH} --no-check-certificate -O ${local_model_save_dir}
fi 

cd ${PROJ_ROOT_PATH}/export_model
if [ ! -d mmcv ]; then
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv 
    git checkout -b v1.7.0 v1.7.0
fi

cd ${PROJ_ROOT_PATH}/export_model
if [ ! -d mmpose ]; then
    git clone https://github.com/open-mmlab/mmpose.git
    cd mmpose
    git checkout -b v0.29.0 v0.29.0
    cd tools
    if [ ! -d data ]; then
        mkdir data
        ln -sf $COCO_DATASETS_PATH data/coco
    fi
fi

cd ${PROJ_ROOT_PATH}/gen_model
if [ ! -d data ]; then
    mkdir data
    ln -sf $COCO_DATASETS_PATH data/coco
fi

file=${PROJ_ROOT_PATH}/export_model/mmpose/tools/deployment/pytorch2onnx.py
str="dynamic_axes"
cd ${PROJ_ROOT_PATH}/export_model/mmpose
if [ `grep -c "$str" $file` -ne '0' ]; then
    echo "patch has been used"
else
    git apply ../magicmind.patch

    # install mmcv
    cd ${PROJ_ROOT_PATH}/export_model/mmcv
    export MMCV_WITH_OPS=1
    export FORCE_MLU=1
    pip install -v -e .

    # install mmpose
    cd $PROJ_ROOT_PATH/export_model/mmpose
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
    pip install -v -e .
fi
