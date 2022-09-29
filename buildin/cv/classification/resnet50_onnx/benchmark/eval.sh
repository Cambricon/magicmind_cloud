#!/bin/bash
set -e
set -x
QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH=$3
COMPUTE_TOP1_AND_TOP5(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    BATCH=$3
    python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_labels.txt \
                                                --result_1_file $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_result_1.txt \
                                                --result_5_file $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_result_5.txt \
                                                --top1andtop5_file $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_result.txt
}

if [ $# != 0 ];
then 
    COMPUTE_TOP1_AND_TOP5 ${QUANT_MODE} ${SHAPE_MUTABLE} ${BATCH}
else  
    echo "Parm doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    #dynamic
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
	cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode true 1	
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $quant_mode true 1 $batch 1000
            COMPUTE_TOP1_AND_TOP5 $quant_mode true $batch
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric top1andtop5 --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_${quant_mode}_true_$batch/eval_result.txt --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_python_output_${quant_mode}_true_${batch}_eval_result.txt --model resnet50_onnx
        done
    done
fi 
