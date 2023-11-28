#!/bin/bash
set -x
pth_name='ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001-b9172d0c.pth'
pth_path=${MODEL_PATH}/${pth_name}
onnx_path=${MODEL_PATH}/ocrnet.onnx

# 拉取mmseg仓库
if [ ! -d 'mmsegmentation' ];then
    git clone https://github.com/open-mmlab/mmsegmentation.git
fi
# 安装mmseg
pushd mmsegmentation
    pip show mmsegmentation
    if [ $? -eq 1 ];then
        # 切换分支
        git checkout -b v0.30.0 v0.30.0
				git apply ../magicmind.patch
        # 安装mmseg
        python setup.py install
    fi
    # 安装依赖
    pip install mmcv-full cityscapesscripts
    # 下载预训练权重
    if [ ! -f ${pth_path} ];then
        wget -P ${MODEL_PATH} \
        https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes/${pth_name}
    fi
    # 生成onnx模型
    if [ ! -f ${onnx_path} ];then
        python tools/pytorch2onnx.py configs/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes.py \
        --checkpoint ${pth_path} \
        --output-file ${onnx_path} \
        --verify
    fi
popd

# 检查数据集
if [ ! -d ${CITYSCAPES_DATASETS_PATH} ];then
    echo ${CITYSCAPES_DATASETS_PATH}" not exist, please download cityscapes datasets. https://www.cityscapes-dataset.com/downloads/"
    exit 1
fi
