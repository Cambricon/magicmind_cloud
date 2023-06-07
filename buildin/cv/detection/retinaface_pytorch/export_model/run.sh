#!/bin/bash
set -e
if [ -z $1 ];
then
    batch_size=1
else
    batch_size=$1
fi

# 1.下载数据集和权重文件
bash get_datasets_and_models.sh

# 2.下载Retinaface源码
cd ${PROJ_ROOT_PATH}/export_model
if [ -d "Pytorch_Retinaface" ];
then
    echo "Pytorch_Retinaface already exists."
    cd Pytorch_Retinaface
else
    echo 'git clone Pytorch_Retinaface'
    git clone https://github.com/biubug6/Pytorch_Retinaface.git
    cd Pytorch_Retinaface
    git checkout b984b4b775b2c4dced95c1eadd195a5c7d32a60b
fi
cd ..

# 3.patch
if grep -q "leaky \= 0.0" Pytorch_Retinaface/models/net.py;
then
    echo "retinaface has been already patched."
else
    echo "patching retinaface..."
    git apply retinaface.diff
fi

# 4.trace model
echo "export model begin"
cd ${PROJ_ROOT_PATH}/export_model/Pytorch_Retinaface
python ../export.py --weights ${PROJ_ROOT_PATH}/data/models/Mobilenet0.25_Final.pth --imgsz 672 1024 --batch_size ${batch_size}
echo "export model end, traced model for retinaface saved in ${PROJ_ROOT_PATH}/data/models/"

