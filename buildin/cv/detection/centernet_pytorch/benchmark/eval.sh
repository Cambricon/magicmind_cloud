#!/bin/bash
set -e
set -x

COMPUTE_COCO(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    LANGUAGES=$4
    IMG_NUM=$5
    OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/${LANGUAGES}_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}

    python $UTILS_PATH/compute_coco_mAP.py --file_list $UTILS_PATH/coco_file_list_5000.txt \
                                           --result_dir $OUTPUT_DIR \
                                           --ann_dir $DATASETS_PATH \
                                           --data_type val2017 \
                                           --json_name $OUTPUT_DIR \
                                           --image_num ${IMG_NUM} 2>&1 |tee $OUTPUT_DIR/log_eval
}


cd $PROJ_ROOT_PATH/export_model
for batch in 1
do
  bash run.sh $batch
done
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for shape_mutable in true
    do
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $precision $shape_mutable $batch 
            cd $PROJ_ROOT_PATH/infer_cpp
            # image_num = 999
            bash run.sh $precision $shape_mutable $batch 1000
            # language = infer_cpp
            COMPUTE_COCO $precision $shape_mutable $batch infer_cpp 1000
        done
    done
done

