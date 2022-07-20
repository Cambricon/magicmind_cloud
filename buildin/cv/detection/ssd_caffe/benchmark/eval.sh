#!/bin/bash
set -e
set -x

QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH=$3
WAYS=$4
COMPUTE_VOC(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    BATCH=$3
    WAYS=$4
    python $UTILS_PATH/compute_voc_mAP.py --path $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/voc_preds/ \
                                          --devkit_path $DATASETS_PATH/VOCdevkit \
                                          --year 2012 2>&1 |tee $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/voc_preds/log_eval
}

if [ $# != 0 ];
then 
    COMPUTE_VOC ${QUANT_MODE} ${SHAPE_MUTABLE} ${BATCH} ${WAYS}
else  
    echo "Parm doesn't exist, run benchmark"
    # dynamic
    cd $PROJ_ROOT_PATH/export_model
    bash get_datasets_and_models.sh
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do 
	cd $PROJ_ROOT_PATH/gen_model
	bash run.sh $quant_mode true 1
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $quant_mode true 1 $batch
            COMPUTE_VOC $quant_mode true $batch infer_python
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric vocmAP --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_${quant_mode}_true_${batch}/voc_preds/log_eval --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_python_output_${quant_mode}_true_${batch}_log_eval --model ssd_caffe
        done
    done
fi
