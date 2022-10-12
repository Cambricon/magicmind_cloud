#!/bin/bash
set -e
set -x
QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH=$3
IMAGE_NUM=$4

#dynamic
if [ $# != 0 ];
then 
    cd $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface/widerface_evaluate
    python3 setup.py build_ext --inplace
    python3 evaluation.py -p $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/pred_txts \
                          -g $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface/widerface_evaluate/ground_truth
else  
    echo "Parm doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode true 1 0.001 0.65 1000
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_cpp
            bash run.sh $quant_mode true 1 $batch -1
            cd $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface/widerface_evaluate
            python3 setup.py build_ext --inplace
            python3 evaluation.py -p $PROJ_ROOT_PATH/data/output/infer_cpp_output_${quant_mode}_true_${batch}/pred_txts \
                                  -g $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface/widerface_evaluate/ground_truth
        done
    done
fi
