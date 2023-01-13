#!/bin/bash
set -e
set -x
COMPUTE_COCO(){
    PRECISION=$1
    BATCH=$2
    python $UTILS_PATH/compute_coco_keypoints.py --ann_file $DATASETS_PATH/annotations/person_keypoints_val2017.json \
                                           --res_file $PROJ_ROOT_PATH/data/images/body25_${PRECISION}_${BATCH}/BODY_25 \
                                           --res2_file $PROJ_ROOT_PATH/data/images/coco_${PRECISION}_${BATCH}/COCO \
                                           --output_file $PROJ_ROOT_PATH/data/images/${PRECISION}_${BATCH}_eval
} 

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision 1
    for batch in 1
    do
        cd $PROJ_ROOT_PATH/infer_cpp
        bash run.sh $precision $batch
        COMPUTE_COCO $precision $batch
    done
done
