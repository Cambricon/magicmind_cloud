#!/bin/bash
set -e
set -x

if [ -d $PROJ_ROOT_PATH/data ];
then
    echo "folder $PROJ_ROOT_PATH/data already exist!!!"
else
    mkdir -p $PROJ_ROOT_PATH/data
fi

if [ -d $PROJ_ROOT_PATH/data/output ];
then
    echo "folder $PROJ_ROOT_PATH/data/output already exist!!!"
else
    mkdir -p $PROJ_ROOT_PATH/data/output
fi

if [ -d $MODEL_PATH ];
then
    echo "folder $MODEL_PATH already exist!!!"
else
    mkdir -p $MODEL_PATH
fi

# 1.下载数据集和模型
bash get_datasets_and_models.sh

# 2.下载yolov5实现源码，切换到v7.0分支
cd $PROJ_ROOT_PATH/export_model
if [ -d "yolov5" ];
then
  echo "yolov5 already exists."
else
  echo "git clone yolov5..."
  git clone https://github.com/ultralytics/yolov5.git
  cd yolov5
  git config --global --add safe.directory $PWD
  git checkout -b v7.0 v7.0
fi

# 3.patch-yolov5
if grep -q "MODEL_PATH" $PROJ_ROOT_PATH/export_model/yolov5/export.py;
then 
  echo "modifying the yolov5s has been already done."
else
  echo "modifying the yolov5s..."
  cd $PROJ_ROOT_PATH/export_model/yolov5
  git apply $PROJ_ROOT_PATH/export_model/yolov5_v7_0_pytorch.patch
fi

# 4.patch-torch-cocodataset
if grep -q "torch.sigmoid(x)" /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py;then
    echo "SiLU activation operator already converted.";
else
    echo "replace SiLU op in '/usr/lib/python3.7/site-packages/torch/nn/modules/activation.py'"
    patch -p0 -f /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < ${PROJ_ROOT_PATH}/export_model/activation.patch
fi

# 5.trace model
if [ ! -f ${MODEL_PATH}/yolov5s_traced.pt ];then
    echo "export model begin..."
    python $PROJ_ROOT_PATH/export_model/yolov5/export.py --weights $MODEL_PATH/yolov5s.pt --imgsz 640 640 --include torchscript --batch-size 1
    echo "export model end..."
else
    echo "trace model already exists."
fi
