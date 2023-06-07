#!/bin/bash
set -e
set -x

if [ -d $MODEL_PATH ];
then
    echo "folder ${MODEL_PATH} already exist!!!"
else
    mkdir "${MODEL_PATH}"
fi

# 1.下载数据集和模型
bash get_datasets_and_models.sh

# 2.下载crnn实现源码
cd ${PROJ_ROOT_PATH}/export_model
if [ -d "crnn.pytorch" ];
then
  echo "crnn.pytorch already exists."
else
  echo "git clone crnn..."
  git clone https://github.com/meijieru/crnn.pytorch.git
  cd crnn.pytorch
  git reset --hard cdf07cc6d8dce0557e542e6cdc0558bd1ad66b53
fi

# 3.patch
if grep -q "trace_model"  ${PROJ_ROOT_PATH}/export_model/crnn.pytorch/demo.py;
then
  echo "patch already be used"
else
  cd ${PROJ_ROOT_PATH}/export_model
  patch -p0 crnn.pytorch/demo.py < demo.patch 
fi

# 4.trace model
if [ -f ${MODEL_PATH}/crnn.pt ];
then
  echo "crnn.pt already exists."
else
  cd ${PROJ_ROOT_PATH}/export_model/crnn.pytorch
  echo "export model begin..."
  python demo.py
  echo "export model end..."
fi
