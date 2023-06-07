#!/bin/bash
set -e
set -x

if [ ! -d ${TOTAL_TEXT_DATASETS_PATH} ]
then
  echo "Please download datasets from https://drive.google.com/drive/folders/12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7 and then unzip -o total_text.zip"
  exit 1
fi
if [ ! -f ${TOTAL_TEXT_DATASETS_PATH}/totaltext.zip ];
then
  echo "Please download test_images from https://drive.google.com/uc?id=1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2&export=download"
  exit 1
fi

if [ ! -d ${TOTAL_TEXT_DATASETS_PATH}/total_text/test_images ];
then
  cd ${TOTAL_TEXT_DATASETS_PATH}
  unzip -o totaltext.zip
  mkdir ${TOTAL_TEXT_DATASETS_PATH}/total_text/test_images
  mv ${TOTAL_TEXT_DATASETS_PATH}/Images/Test/* ${TOTAL_TEXT_DATASETS_PATH}/total_text/test_images/
fi

if [ ! -d ${PROJ_ROOT_PATH}/export_model/DB/datasets ];
then
    mkdir ${PROJ_ROOT_PATH}/export_model/DB/datasets
    cp -r ${TOTAL_TEXT_DATASETS_PATH}/total_text ${PROJ_ROOT_PATH}/export_model/DB/datasets
fi

if [ ! -d ${PROJ_ROOT_PATH}/data ];
then
    mkdir -p ${PROJ_ROOT_PATH}/data
fi
if [ ! -d ${MODEL_PATH} ];
then
    mkdir -p ${MODEL_PATH}
fi
if [ ! -f ${MODEL_PATH}/totaltext_resnet18 ];
then
    echo "Please download  the converted ground-truth and data list from https://drive.google.com/drive/folders/1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG"
    exit 1
fi
