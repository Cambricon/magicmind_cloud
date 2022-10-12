#!/bin/bash
set -e
set -x

QUANT_MODE=$1
SHAPE_MUTABLE=$2
THREADS=$3
INPUT_SIZE=$4
MM_RUN(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    THREADS=$3
    INPUT_SIZE=$4
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $MODEL_PATH/deeplabv3_tf_model_${QUANT_MODE}_${SHAPE_MUTABLE} \
                          --threads ${THREADS} \
                          --input_dims 1,${INPUT_SIZE},${INPUT_SIZE},3 \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${INPUT_SIZE}_log_perf
}

if [ $# != 0 ];
then
    MM_RUN ${QUANT_MODE} ${SHAPE_MUTABLE} ${THREADS} ${INPUT_SIZE}
else
    echo "Parm Doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        for shape_mutable in true
        do 
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $quant_mode $shape_mutable
            for input_size in 513
            do
	        for thread in 1  
                do
                    MM_RUN $quant_mode $shape_mutable $thread $input_size
                    python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${input_size}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${shape_mutable}_${input_size}_log_perf --model deeplabv3_tf
                done
            done
        done
	for shape_mutable in false
        do
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $quant_mode $shape_mutable
            for input_size in 513
            do
                for thread in 1
                do
                    MM_RUN $quant_mode $shape_mutable $thread $input_size
                    python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${input_size}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${shape_mutable}_${input_size}_log_perf --model deeplabv3_tf
                done
            done
        done
    done
fi
