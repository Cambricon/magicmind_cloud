#!/bin/bash
set -e
set -x
BATCH_SIZE=$1

if [ -d ${MODEL_PATH} ];then
    echo "folder ${MODEL_PATH} already exist!!!"
else
    mkdir -p "${MODEL_PATH}"
fi

# 1.下载数据集和模型
bash get_datasets_and_models.sh

# 2.下载yolov7实现源码
cd ${PROJ_ROOT_PATH}/export_model
if [ -d "yolov7" ];then
    echo "yolov7 already exists."
else
    echo "git clone yolov7..."
    git clone https://github.com/WongKinYiu/yolov7.git
    cd yolov7
    git checkout 072f76c72c641c7a1ee482e39f604f6f8ef7ee92
fi

# 3.patch-yolov7
if [ ! -f ${PROJ_ROOT_PATH}/export_model/yolov7/gen_tracemodel.py ];then 
    echo "modifying the yolov7..."
    cd ${PROJ_ROOT_PATH}/export_model/yolov7
    git apply ${PROJ_ROOT_PATH}/export_model/yolov7_pytorch.patch
else
    echo "patch already applied!"
fi

# 4.patch-torch
if grep -q "torch.sigmoid(x)" /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py;then
    echo "SiLU activation operator already converted.";
else
    echo "replace SiLU op in '/usr/lib/python3.7/site-packages/torch/nn/modules/activation.py'"
    patch -p0 -f /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < ${PROJ_ROOT_PATH}/export_model/activation.patch
fi

# 5.export model
if [ ! -f ${MODEL_PATH}/yolov7_traced_model.pt ];then
    cd ${PROJ_ROOT_PATH}/export_model/yolov7 
    echo "export model begin..."
    python ${PROJ_ROOT_PATH}/export_model/yolov7/gen_tracemodel.py --weight ${MODEL_PATH}/yolov7.pt --save_path ${MODEL_PATH}/yolov7_traced_model.pt
    echo "export model end..."
fi
