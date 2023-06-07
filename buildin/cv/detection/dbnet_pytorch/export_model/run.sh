#!/bin/bash
set -x
set -e

echo "*******************************************"
echo "          export model begin               "
echo "*******************************************"

#1. clone the dbnet github
if [ ! -d ${PROJ_ROOT_PATH}/export_model/DB ];
then
  git clone https://github.com/MhLiao/DB.git
  cd DB
  git reset --hard e5a12f5c2f0c2b4a345b5b8392307ef73481d5f6
fi

#2. download datasets and models
cd ${PROJ_ROOT_PATH}/export_model
bash get_datasets_and_models.sh

#3. dcn
if [ ! -d ${PROJ_ROOT_PATH}/export_model/DB/assets/ops/cpu_deform_conv ];
then
  cp -r ${PROJ_ROOT_PATH}/export_model/cpu_deform_conv ${PROJ_ROOT_PATH}/export_model/DB/assets/ops
fi
if [ ! -f ${PROJ_ROOT_PATH}/export_model/DB/backbones/vision_deform_conv.py ];
then
  cp ${PROJ_ROOT_PATH}/export_model/vision_deform_conv.py ${PROJ_ROOT_PATH}/export_model/DB/backbones/
fi

#4. patch
if grep -q "vision_deform_conv"  ${PROJ_ROOT_PATH}/export_model/DB/backbones/resnet.py;
then
  echo "backbones_resnet.patch already be used"
else
  cd ${PROJ_ROOT_PATH}/export_model
  patch -p0 ${PROJ_ROOT_PATH}/export_model/DB/backbones/resnet.py < backbones_resnet.patch
fi

#5. setup
if [ ! -f ${PROJ_ROOT_PATH}/export_model/DB/assets/ops/cpu_deform_conv/torchvision_cpu_dcn.so ];
then
  cd ${PROJ_ROOT_PATH}/export_model/DB/assets/ops/cpu_deform_conv/
  python setup.py build develop
fi

#6. save dbnet.pt
if [ ! -f "${MODEL_PATH}/dbnet.pt" ];
then
  if grep -q "dbnet.pt"  ${PROJ_ROOT_PATH}/export_model/DB/structure/model.py;
  then
    echo "structure_model.patch already be used"
  else
    cd ${PROJ_ROOT_PATH}/export_model
    patch -p0 ${PROJ_ROOT_PATH}/export_model/DB/structure/model.py < structure_model.patch
  fi

  cd ${PROJ_ROOT_PATH}/export_model/DB
  python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --image_path ${TOTAL_TEXT_DATASETS_PATH}/total_text/test_images/img1.jpg --resume ${MODEL_PATH}/totaltext_resnet18 --polygon --box_thresh 0.7 --visualize
  git checkout ${PROJ_ROOT_PATH}/export_model/DB/structure/model.py
  patch -p0 ${PROJ_ROOT_PATH}/export_model/DB/structure/model.py < ${PROJ_ROOT_PATH}/export_model/dcn.patch
fi
