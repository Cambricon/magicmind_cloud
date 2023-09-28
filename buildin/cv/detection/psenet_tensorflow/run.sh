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

###3.infer_python
cd $PROJ_ROOT_PATH/infer_python
bash run.sh force_float32 false 500

echo "INFER SUCCESS!"

###4.compute accuracy
THIS_OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/psenet_tf_result_force_float32_false_1.json
python $UTILS_PATH/compute_icdar_hmean.py   --label_file  $ICDAR_DATASETS_PATH/icdar2015/icdar2015_test_label.json \
                                            --result_dir $THIS_OUTPUT_DIR  \
                                            2>&1 | tee $PROJ_ROOT_PATH/data/output/psenet_tf_force_float32_false_1_log_eval
echo "All has benn Finish!"
