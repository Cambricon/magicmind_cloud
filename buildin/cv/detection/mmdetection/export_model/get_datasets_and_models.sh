#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"

if [ ! -d ${COCO_DATASETS_PATH} ];then
  mkdir -p ${COCO_DATASETS_PATH}
fi
cd ${COCO_DATASETS_PATH}

if [ ! -d ${FILE1} ];then 
  echo "Downloading val2017.zip"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -o val2017.zip
fi

if [ ! -d ${FILE2} ];then
  echo "Downloading annotations_trainval2017.zip"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -o annotations_trainval2017.zip
fi

if [ ! -f ${FILE3} ];then
  echo "coco.names"
  wget -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names --no-check-certificate -O coco.names
fi 

if [ ! -d ${MODEL_PATH} ];then
    mkdir -p ${MODEL_PATH}
fi
cd ${MODEL_PATH}

splits=(${MMDETECTION_MODEL_PRETRAINED_PATH//// })
local_model_save_dir=${splits[-1]}
if [ ! -f ${MODEL_PATH}/${local_model_save_dir} ];then
  wget -c ${MMDETECTION_MODEL_PRETRAINED_PATH} --no-check-certificate -O ${local_model_save_dir}
fi 

cd ${PROJ_ROOT_PATH}/export_model
if [ ! -d "mmdetection" ];then
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    # v2.28.2
    git checkout e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1

else
    echo "mmdetection exist!"
fi

cd $PROJ_ROOT_PATH/export_model/mmdetection
if grep -q "COCO_DATASETS_PATH" ${PROJ_ROOT_PATH}/export_model/mmdetection/configs/_base_/datasets/coco_instance.py;then
    echo "patch has been patched.";
else  
    git apply ../magicmind.patch
fi
pip install mmcv-full==1.6.0
pip install -e .

