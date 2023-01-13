#!/bin/bash
set -e
set -x
precision=force_float32
shape_mutable=true
batch_size=1
conf=0.001
iou=0.65
max_det=300
image_num=1000
languages=infer_python

# 0. export model
cd $PROJ_ROOT_PATH/export_model 
bash run.sh

# 1. gen model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision $shape_mutable $batch_size $conf $iou $max_det

# 2 infer
if [ $languages == "infer_cpp" ];
then
  cd $PROJ_ROOT_PATH/infer_cpp
  bash run.sh $precision $shape_mutable $image_num
fi
if [ $languages == "infer_python" ];
then
  cd $PROJ_ROOT_PATH/infer_python
  bash run.sh $precision $shape_mutable $image_num
fi

# 3.eval
python $UTILS_PATH/compute_coco_mAP.py  --file_list $UTILS_PATH/coco_file_list_5000.txt \
                                        --result_dir $PROJ_ROOT_PATH/data/output/${languages}_output_${precision}_${shape_mutable}_1 \
                                        --ann_dir $DATASETS_PATH \
                                        --data_type val2017 \
                                        --json_name $PROJ_ROOT_PATH/data/output/yolov5_${precision}_${shape_mutable}_1.json \
                                        --language $languages \
                                        --image_num $image_num 2>&1 |tee $PROJ_ROOT_PATH/data/output/${languages}_output_${precision}_${shape_mutable}_1/log_eval
