#!/bin/bash
set -e
set -x

magicmind_model=${1}
batch_size=${2}
image_num=${3}
yolo_mode=${4:-val}

if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
  mkdir "$PROJ_ROOT_PATH/data/output"
fi

echo "infer Magicmind model..."
cd ${PROJ_ROOT_PATH}/export_model/ultralytics

predict_dataset_dir="${PROJ_ROOT_PATH}/export_model/ultralytics/datasets/coco"
if [ ! -d "$predict_dataset_dir" ];
then
  mkdir -p ${PROJ_ROOT_PATH}/export_model/ultralytics/datasets/
fi


# validate coco dataset
if [ ${yolo_mode} == "val" ];
then
  if [ ! -L ${predict_dataset_dir} ];
  then
    ln -sf ${COCO_DATASETS_PATH} ${predict_dataset_dir}
  fi
  yolo task=detect mode=val model=${MODEL_PATH}/yolov8n.pt data=${PROJ_ROOT_PATH}/export_model/ultralytics/ultralytics/cfg/datasets/coco.yaml mm_model=${magicmind_model} batch=${batch_size}
fi

# predict given dataset
if [ ${yolo_mode} == "predict" ];
then
  # run the val mode before and made a soft link
  if [ -L ${predict_dataset_dir} ];
  then
    rm ${predict_dataset_dir}
  fi
  # dataset needs to be copied before running the predict mode
  if [ ! -d ${predict_dataset_dir} ];
  then
    mkdir -p ${predict_dataset_dir}
    cp -r ${COCO_DATASETS_PATH}/* ${predict_dataset_dir}
  fi

  infer_res_dir="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_infer_res"
  if [ ! -d "$infer_res_dir" ];
  then
    mkdir "$infer_res_dir"
  fi

  head -n ${image_num} ${COCO_DATASETS_PATH}/val2017.txt > ${predict_dataset_dir}/val2017_${image_num}.txt
  yolo task=detect mode=predict model=${MODEL_PATH}/yolov8n.pt source=${predict_dataset_dir}/val2017_${image_num}.txt mm_model=${magicmind_model}
  echo "Results saved to ${PROJ_ROOT_PATH}/export_model/ultralytics/runs/detect/"
fi
