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
bash run.sh force_float32 false 1

echo "GENERATE MODEL SUCCESS!"

###3.infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh force_float32 false 5000

echo "INFER CPP SUCCESS!"

###4.compute accuracy
THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_false_1"
python $UTILS_PATH/compute_coco_mAP.py  --file_list  $UTILS_PATH/coco_file_list_5000.txt \
                                        --result_dir $THIS_OUTPUT_DIR/  \
                                        --ann_dir $DATASETS_PATH/ \
                                        --data_type val2017 \
                                        --json_name $PROJ_ROOT_PATH/data/output/yolov3_tf_force_float32_false_1 \
                                        --img_dir $DATASETS_PATH/val2017 \
                                        --image_num 5000 2>&1 | tee $PROJ_ROOT_PATH/data/output/yolov3_tf_force_float32_false_1_log_eval
echo "All has benn Finish!"
