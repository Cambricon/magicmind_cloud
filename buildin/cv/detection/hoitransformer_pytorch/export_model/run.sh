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

# 2.下载HoiTransformer实现源码
cd $PROJ_ROOT_PATH/export_model
if [ -d "HoiTransformer" ];
then
  echo "HoiTransformer already exists."
else
  echo "git clone HoiTransformer..."
  git clone https://github.com/bbepoch/HoiTransformer.git
  cd HoiTransformer
  git reset --hard 49f0573a2ec46357ff38661b55db4cb43cdaaad6
fi

# 3.patch-HoiTransformer
if grep -q "reshape"  $PROJ_ROOT_PATH/export_model/HoiTransformer/models/position_encoding.py;
then
  echo "modifying the HoiTransformer has been already done"
else
  echo "modifying the Hoitransformer..."
  cd $PROJ_ROOT_PATH/export_model/HoiTransformer
  git apply $PROJ_ROOT_PATH/export_model/model.patch
fi

# 4.export onnx model
cd $PROJ_ROOT_PATH/export_model
echo "export model begin..."
export PYTHONPATH=$PROJ_ROOT_PATH/export_model/HoiTransformer:$PYTHONPATH
python3 export_onnx.py --dataset_file=hoia --backbone=resnet50 --batch_size=1 --model_path=$MODEL_PATH/res50_hoia_a4caffe.pth
echo "export model end..."
