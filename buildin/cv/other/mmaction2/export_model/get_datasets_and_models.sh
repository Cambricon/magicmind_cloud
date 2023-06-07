#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"

if [ ! -d ${KINETICS_POSTPROCESS_DATASETS_PATH} ];then
    echo ${KINETICS_POSTPROCESS_DATASETS_PATH}" not exist, please download kinetics datasets. https://www.deepmind.com/open-source/kinetics/"    
    exit 1
fi

if [ ! -d ${MODEL_PATH} ];then
    mkdir -p ${MODEL_PATH}
fi
cd ${MODEL_PATH}

splits=(${MMACTION2_MODEL_PRETRAINED_PATH//// })
local_model_save_dir=${splits[-1]}
if [ ! -f ${MODEL_PATH}/${local_model_save_dir} ];then
  wget -c ${MMACTION2_MODEL_PRETRAINED_PATH} --no-check-certificate -O ${local_model_save_dir}
fi 

cd ${PROJ_ROOT_PATH}/export_model
if [ ! -d "mmaction2" ];then
    git clone https://github.com/open-mmlab/mmaction2.git
    cd mmaction2
    # 切换分支
    git checkout -b v0.24.1 v0.24.1
else
    echo "mmaction2 exist!"
fi

cd $PROJ_ROOT_PATH/export_model/mmaction2
if grep -q "KINETICS_POSTPROCESS_DATASETS_PATH" ${PROJ_ROOT_PATH}/export_model/mmaction2/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py;then
    echo "patch has been patched."
else  
    git apply ../magicmind.patch
    # 安装依赖
    # pip install mmengine
    pip install mmcv==1.7.0
    pip install mmcv-full==1.7.0
    pip install mmdet==2.28.2
    # #安装mmaction2
    pip install -v -e .
fi

