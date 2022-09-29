#!/bin/bash
set -e
set -x
QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH=$3
WAYS=$4

COMPUTE_VOC_MIOU(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    BATCH=$3
    WAYS=$4
    python $UTILS_PATH/compute_voc_mIOU_segnet.py --output_dir $PROJ_ROOT_PATH/data/output/${WAYS}_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}
}

# dynamic
if [ $# != 0 ];
then 
    COMPUTE_VOC_MIOU ${QUANT_MODE} ${SHAPE_MUTABLE} ${BATCH} ${WAYS}
else  
    echo "Parm doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
	cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode true 1	
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_cpp
            bash run.sh $quant_mode true 1 $batch 1000
            COMPUTE_VOC_MIOU $quant_mode true $batch infer_cpp
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric voc_miou --output_file $PROJ_ROOT_PATH/benchmark/benchmark.csv --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_cpp_output_${quant_mode}_true_${batch}_benchmark_ok.csv --model segnet_bn_caffe
        done
    done
fi
