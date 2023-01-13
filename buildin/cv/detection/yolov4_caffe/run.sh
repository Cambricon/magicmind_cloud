#!/bin/bash
set -e
set -x
echo "Start !"

image_num=1000

source env.sh
###1.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh
echo "DOWNLOAD_DATA_SUCCESS!"

###2.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 false 1

echo "GENERATE MODEL SUCCESS!"

###3.infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh force_float32 false $image_num

echo "INFER CPP SUCCESS!"

###4.compute mAP
python $UTILS_PATH/compute_coco_mAP.py  --file_list $UTILS_PATH/coco_file_list_5000.txt \
                                        --result_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_false_1 \
                                        --ann_dir $DATASETS_PATH/ \
                                        --data_type val2017 \
                                        --json_name $PROJ_ROOT_PATH/data/output/yolov4_force_float32_false_1 \
                                        --img_dir $DATASETS_PATH/val2017 \
                                        --image_num $image_num 2>&1 | tee $PROJ_ROOT_PATH/data/output/yolov4_force_float32_false_1_log_eval

