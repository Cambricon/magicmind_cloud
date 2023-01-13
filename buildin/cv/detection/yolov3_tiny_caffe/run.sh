#!/bin/bash
set -e
set -x
echo "Start !"

source env.sh
###1.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh
echo "DOWNLOAD_DATA_SUCCESS!"

###2.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
echo "GENERATE MODEL SUCCESS!"

###3.infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
echo "INFER CPP SUCCESS!"

###4.compute acc
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/force_float32_true_1
python $UTILS_PATH/compute_coco_mAP.py  --file_list ${OUTPUT_DIR}/json/image_name.txt \
                                        --result_dir $OUTPUT_DIR/results \
                                        --ann_dir $DATASETS_PATH/ \
                                        --data_type 'val2017' \
                                        --json_name $OUTPUT_DIR/json/force_float32_true_1 \
                                        --img_dir $DATASETS_PATH/val2017 \
                                        --image_num 1000
echo "All has benn Finish!"
