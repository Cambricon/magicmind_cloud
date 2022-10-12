#!/bin/bash
set -e
set -x

QUANT_MODE=$1
BATCH_SIZE=$2

COMPUTE_ACCURACY(){
    QUANT_MODE=$1
    BATCH_SIZE=$2
    python ${UTILS_PATH}/ijbc_eval.py --features_dir ${PROJ_ROOT_PATH}/data/images/${QUANT_MODE}_${BATCH_SIZE} \
	                              --output_file $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${BATCH_SIZE} \
				      --face_tid_mid_file $PROJ_ROOT_PATH/infer_cpp/1000_face_tid_mid.txt \
                                      --template_pair_label_file ${PROJ_ROOT_PATH}/infer_cpp/1000_template_pair_label.txt
}
# dynamic
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi

if [ $# != 0 ];
then
    COMPUTE_ACCURACY ${QUANT_MODE} ${BATCH_SIZE}
else
    echo "Parm doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    cd $PROJ_ROOT_PATH/infer_cpp
    bash build.sh
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode 1	
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_cpp
            bash run.sh $quant_mode $batch
            COMPUTE_ACCURACY $quant_mode $batch
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric 1e5and1e4 --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${batch} --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${batch} --model $MODEL_NAME
        done
    done
fi
