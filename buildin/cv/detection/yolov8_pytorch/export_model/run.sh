#!/bin/bash
set -e
set -x

batch_size=${1}

if [ -d ${MODEL_PATH} ];then
    echo "folder ${MODEL_PATH} already exist!!!"
else
    mkdir -p "${MODEL_PATH}"
fi

# 1.下载数据集和模型
bash get_datasets_and_models.sh

# 2.下载yolov8源码
cd ${PROJ_ROOT_PATH}/export_model
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
if [ -d "ultralytics" ];then
    echo "ultralytics already exists."
else
    echo "git clone ultralytics..."
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics
    git checkout bcec003ee30a181403d223757f424a50a86cd426
fi

# 3.patch-torch
if grep -q "F.silu" /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py;
then
  echo "add SiLU op in '/usr/lib/python3.7/site-packages/torch/nn/modules/activation.py'"
  patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < ${PROJ_ROOT_PATH}/export_model/activation.patch
else
  echo "SiLU activation operator already exists.";
fi

# 4.patch yolov8
if grep -q "MMRunner" $PROJ_ROOT_PATH/export_model/ultralytics/ultralytics/engine/validator.py;
then
  echo "ultralytics has been already patched"
else
  echo "modifying the ultralytics..."
  cd $PROJ_ROOT_PATH/export_model/ultralytics/
  git apply $PROJ_ROOT_PATH/export_model/yolov8_pytorch.patch
fi

# 5.export model
if [ ! -f ${MODEL_PATH}/yolov8n.onnx ];then
    echo "export model begin..."
    cd $PROJ_ROOT_PATH/export_model/ultralytics/
    python setup.py install
    yolo task=detect mode=export model=${MODEL_PATH}/yolov8n.pt format=onnx source=${PROJ_ROOT_PATH}/export_model/ultralytics/assets/bus.jpg opset=11 dynamic=true simplify=true
    echo "export model end..."
fi
