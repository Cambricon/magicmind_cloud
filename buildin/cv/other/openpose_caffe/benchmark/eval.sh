#!/bin/bash
set -e
set -x
QUANT_MODE=$1
BATCH=$2
COMPUTE_COCO(){
    QUANT_MODE=$1
    BATCH=$2
    python $UTILS_PATH/compute_coco_keypoints.py --ann_file $DATASETS_PATH/annotations/person_keypoints_val2017.json \
                                           --res_file $PROJ_ROOT_PATH/data/images/body25_${QUANT_MODE}_${BATCH}/BODY_25 \
                                           --res2_file $PROJ_ROOT_PATH/data/images/coco_${QUANT_MODE}_${BATCH}/COCO \
                                           --output_file $PROJ_ROOT_PATH/data/images/${QUANT_MODE}_${BATCH}_eval
} 

if [ $# != 0 ];
then 
    COMPUTE_COCO ${QUANT_MODE} ${BATCH}
else  
    echo "Parm doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode 1
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_cpp
            bash run.sh $quant_mode $batch
            COMPUTE_COCO $quant_mode $batch
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric cocoKeyPoints --output_file $PROJ_ROOT_PATH/data/images/${quant_mode}_${batch}_eval --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${batch}_eval --model openpose_caffe 
        done
    done
fi
