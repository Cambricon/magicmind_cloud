#!/bin/bash
set -e

echo "*******************************************"
echo "          export model begin               "
echo "*******************************************"

# 1.下载数据集和模型
bash get_datasets_and_models.sh

# 2.下载yolov5实现源码，切换到v6.1分支
cd ${PROJ_ROOT_PATH}/export_model
if [ -d "yolov5" ];then
    echo "yolov5 already exists."
else
    echo "Start to git clone yolov5..."
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    git checkout -b v6.1 v6.1
fi

# 3.patch-yolov5
if grep -q "yolov5m_traced.pt" ${PROJ_ROOT_PATH}/export_model/yolov5/export.py;then 
    echo "Modifying the yolov5m has been already done"
else
    echo "Modifying the yolov5m..."
    cd ${PROJ_ROOT_PATH}/export_model/yolov5
    git apply ${PROJ_ROOT_PATH}/export_model/yolov5_v6_1_pytorch.patch
fi

# 4.patch-torch-cocodataset
if grep -q "SiLU" /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py;then
    echo "SiLU activation operator already exists.";
else
    echo "Add SiLU op in '/usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py and activation.py'"
    patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py < ${PROJ_ROOT_PATH}/export_model/init.patch
    patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < ${PROJ_ROOT_PATH}/export_model/activation.patch
fi

# 5.trace model
traced_pt_file=${MODEL_PATH}/yolov5m_traced.pt
if [ ! -f ${traced_pt_file} ];then
    echo "Start to generate traced pt."
    cd ${PROJ_ROOT_PATH}/export_model 
    python ${PROJ_ROOT_PATH}/export_model/yolov5/export.py --weights ${MODEL_PATH}/yolov5m.pt --imgsz 640 640 --include torchscript --batch-size 1
else
    echo "traced pt already exists!"
fi
