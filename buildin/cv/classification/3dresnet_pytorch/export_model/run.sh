#!/bin/bash
set -e
set -x

mkdir -p ${PROJ_ROOT_PATH}/data/output
mkdir -p ${MODEL_PATH}

# 1.下载数据集和模型
bash get_datasets_and_models.sh

# 2.下载3D-ResNets-PyTorch实现源码
cd ${PROJ_ROOT_PATH}/export_model
if [ -d "3D-ResNets-PyTorch" ];
then
  echo "3D-ResNets-PyTorch already exists."
else
  echo "git clone 3D-ResNets-PyTorch..."
  git clone https://github.com/kenshohara/3D-ResNets-PyTorch.git
  cd 3D-ResNets-PyTorch
  git reset --hard 540a0ea1abaee379fa3651d4d5afbd2d667a1f49 
fi

# 3.patch-3D-ResNets-PyTorch
if grep -q "generate_pt"  ${PROJ_ROOT_PATH}/export_model/3D-ResNets-PyTorch/main.py;
then
  echo "modifying the 3D-ResNets-PyTorch has been already done"
else
  echo "modifying the Hoitransformer..."
  cd ${PROJ_ROOT_PATH}/export_model/3D-ResNets-PyTorch
  git apply ${PROJ_ROOT_PATH}/export_model/model.patch
fi

# 4.trace pt model
cd ${PROJ_ROOT_PATH}/export_model/3D-ResNets-PyTorch
if [ -d "data" ];
then
  echo "data in 3D-ResNets-PyTorch exists."
else
  echo "create data dir in 3D-ResNets-PyTorch..."
  mkdir data
fi
cd data
if [ -d "weights" ];
then
  echo "weights in 3D-ResNets-PyTorch data directory exists."
else
  echo "create weights dir in 3D-ResNets-PyTorch data directory..."
  mkdir weights
fi
if [ -L "kinetics_videos" ];
then
  echo "kinetics_videos in 3D-ResNets-PyTorch data directory exists."
else
  echo "create kinetics_videos link in 3D-ResNets-PyTorch data directory..."
  ln -s ${KINETICS_DATASETS_PATH}/kinetics_videos ./ 
fi
if [ -L "kinetics.json" ];
then
  echo "kinetics.json in 3D-ResNets-PyTorch data directory exists."
else
  echo "create kinetics.json link in 3D-ResNets-PyTorch data directory..."
  ln -s ${KINETICS_DATASETS_PATH}/kinetics_videos/kinetics.json ./
fi
if [ -L "weights/r3d50_K_200ep.pth" ];
then
  echo "r3d50_K_200ep.pth in 3D-ResNets-PyTorch data/weights/ directory exists."
else
  echo "create r3d50_K_200ep.pth link in 3D-ResNets-PyTorch data/weights/ directory..."
  ln -s ${MODEL_PATH}/r3d50_K_200ep.pth weights/ 
fi

cd ..
echo "export model begin..."
export PYTHONPATH=${PROJ_ROOT_PATH}/export_model/3D-ResNets-PyTorch:$PYTHONPATH
python main.py --root_path data \
           --video_path kinetics_videos/jpg \
           --annotation_path kinetics.json \
           --result_path ${PROJ_ROOT_PATH}/data/output \
           --dataset kinetics \
           --resume_path weights/r3d50_K_200ep.pth \
           --model_depth 50 \
           --n_classes 700 \
           --n_threads 4 \
           --no_train \
           --no_val \
           --inference \
           --output_topk 5 \
           --inference_batch_size 1 \
           --no_cuda \
           --generate_pt
echo "export model end..."
