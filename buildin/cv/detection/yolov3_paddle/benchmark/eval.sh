#!/bin/bash
set -e
set -x

COMPUTE_COCO(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    python $UTILS_PATH/compute_coco_keypoints.py --res_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/bbox.json \
                                           --ann_file $COCO_DATASETS_PATH/annotations/instances_val2017.json \
                                           --iou_type bbox 2>&1 |tee $PROJ_ROOT_PATH/data/output/yolov3_paddle_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_eval

}

#dynamic
cd $PROJ_ROOT_PATH/export_model
bash run.sh 
for precision in force_float32 force_float16 qint8_mixed_float16
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision false 1
    for batch in 1
    do
        cd $PROJ_ROOT_PATH/infer_python
        bash run.sh $precision false $batch 5000
        COMPUTE_COCO $precision false $batch
    done
done
