#!/bin/bash
set -e
set -x
echo "Start !"

source env.sh
###0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh
echo "DOWNLOAD_DATA_SUCCESS!"

###1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision shape_mutable batch_size
bash run.sh force_float32 false 1

###2.infer
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh precision shape_mutable batch_size image_num
bash run.sh force_float32 false 1 5000

### 3.eval

python $UTILS_PATH/compute_coco_keypoints.py --res_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_false_1/bbox.json \
                                           --ann_file $COCO_DATASETS_PATH/annotations/instances_val2017.json \
                                           --iou_type bbox 
