#!/bin/bash
set -e
set -x
COMPUTE_VOC_MIOU(){
    PRECISION=$1
    IMAGE_NUM=$2
    LANGUAGES=$3
    if [ ! -d $PROJ_ROOT_PATH/data/output/ ]; then mkdir -p $PROJ_ROOT_PATH/data/output/; fi
    python $UTILS_PATH/compute_voc_mIOU_eval.py --image_num $IMAGE_NUM \
                                                --language $LANGUAGES \
                                                --pred_dir $PROJ_ROOT_PATH/data/output/${LANGUAGES}_output_${PRECISION} 2>&1 |tee $PROJ_ROOT_PATH/data/output/${LANGUAGES}_${PRECISION}_log_eval
}

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for shape_mutable in true
    do 
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $precision 
        cd $PROJ_ROOT_PATH/infer_cpp
        bash run.sh $precision 1449
        COMPUTE_VOC_MIOU $precision 1449 infer_cpp
    done
done
