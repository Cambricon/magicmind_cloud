#!/bin/bash
set -e
set -x
echo "Start !"

source env.sh
###1.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh
echo "<<<<<<<<<DOWNLOAD_DATA_SUCCESS!>>>>>>>>>>>>>>"

###2.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 false 1

echo "<<<<<<<<<GENERATE MODEL SUCCESS!>>>>>>>>>>>>>>"

###3.infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh force_float32 false 1

echo "<<<<<<<<<INFER PYTHON SUCCESS!>>>>>>>>>>>>>>"

###4.compute performace
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh
echo "<<<<<<<<<TEST PERFORMANCE SUCCESS!>>>>>>>>>>>>>>"

###5.compute accuracy
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
echo "<<<<<<<<<EVAL SUCCESS!>>>>>>>>>>>>>>"

###6.compare eval and perf
python $MAGICMIND_CLOUD/test/compare_eval.py --metric cocomAP --output_file $PROJ_ROOT_PATH/data/output/force_float32_false_1_log_eval --output_ok_file $PROJ_ROOT_PATH/data/output_ok/force_float32_false_1_log_eval --model yolov3_tiny_caffe
python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/force_float32_false_1_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/force_float32_false_1_log_perf --model yolov3_tiny_caffe 
echo "<<<<<<<<<COMPARE SUCCESS!>>>>>>>>>>>>>>"
echo "All has benn Finish!"