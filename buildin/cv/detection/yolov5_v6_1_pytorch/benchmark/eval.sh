#!/bin/bash
set -e
set -x
COMPUTE_COCO(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    IMG_NUM=$4
    LANGUAGES=$5
    python $UTILS_PATH/compute_coco_mAP.py --file_list $UTILS_PATH/coco_file_list_5000.txt \
                                           --result_dir $PROJ_ROOT_PATH/data/output/${LANGUAGES}_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                                           --ann_dir $DATASETS_PATH \
                                           --data_type val2017 \
                                           --json_name $PROJ_ROOT_PATH/data/output/yolov5_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}.json \
                                           --language $LANGUAGES \
                                           --image_num $IMG_NUM 2>&1 |tee $PROJ_ROOT_PATH/data/output/${LANGUAGES}_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/log_eval
}

languages=infer_cpp
image_num=5000
conf=0.001
iou=0.65
max_det=1000
cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for shape_mutable in true
    do
        for batch_size in 1
        do
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $precision $shape_mutable $batch_size $conf $iou $max_det
            cd $PROJ_ROOT_PATH/$languages
            bash run.sh $precision $shape_mutable $image_num
            COMPUTE_COCO $precision $shape_mutable $batch_size $image_num $languages
        done
    done
done