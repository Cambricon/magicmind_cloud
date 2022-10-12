#!/bin/bash
set -e
if [ -z $1 ];
then
    BATCH_SIZE=1
else
    BATCH_SIZE=$1
fi

# 1.下载数据集
./get_datasets.sh

# 2.下载权重文件
mkdir -p $PROJ_ROOT_PATH/data/models
cd $PROJ_ROOT_PATH/data/models
if [ -f "Resnet50_Final.pth" ];
then 
    echo "Resnet50_Final.pth already exists."
else 
    echo "Downloading Resnet50_Final.pth file"
    gdown -c https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW -O Resnet50_Final.pth
fi

# 3.下载Retinaface源码
cd $PROJ_ROOT_PATH/export_model
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

# 4.patch
if grep -q "leaky \= 0.0" Pytorch_Retinaface/models/net.py;
then
    echo "retinaface has been already patched."
else
    echo "patching retinaface..."
    git apply retinaface.diff
fi

# 5.trace model
cd $PROJ_ROOT_PATH/export_model 
set +e
mkdir -p /root/.cache/torch/hub/checkpoints/
cp Pytorch_Retinaface/resnet50-19c8e357.pth /root/.cache/torch/hub/checkpoints/
set -e
echo "export model begin"
python $PROJ_ROOT_PATH/export_model/export.py --weights $PROJ_ROOT_PATH/data/models/Resnet50_Final.pth --imgsz 672 1024 --batch_size $BATCH_SIZE
echo "export model end, traced model for retinaface saved in $PROJ_ROOT_PATH/data/models/"

