#!/bin/bash
set -e
set -x
PARAMETER_ID=$1
QUANT_MODE=$2
SHAPE_MUTABLE=$3
BATCH_SIZE=$4
BATCH=$5
THREADS=$6

MM_RUN(){
    PARAMETER_ID=$1
    QUANT_MODE=$2
    SHAPE_MUTABLE=$3
    BATCH_SIZE=$4
    BATCH=$5
    THREADS=$6
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/magicmind_models/nnUNet_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${PARAMETER_ID} \
                          --iterations 1000 \
			  --input_dims ${BATCH},320,256,1 \
                          --threads ${THREADS} \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}bs_${PARAMETER_ID}_log_perf 
}

###dynamic
if [ $# != 0 ];
then
    MM_RUN ${PARAMETER_ID} ${QUANT_MODE} ${SHAPE_MUTABLE} 1 ${BATCH} ${THREADS}
else
    echo "Parm Doesn't exist, run benchmark"
    for parameter_id in 0
    do
        cd $PROJ_ROOT_PATH/export_model
        bash run.sh $parameter_id 1
        for quant_mode in force_float32 force_float16 qint8_mixed_float16
        do
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $parameter_id $quant_mode true 1
            for batch in 1 4 8
            do
                for thread in 1
                do 
                    MM_RUN $parameter_id $quant_mode true 1 $batch $thread
                    python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_true_${batch}bs_${parameter_id}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_true_${batch}bs_${parameter_id}_log_perf --model nnUNet_pytorch
                done
            done
        done
    done
fi
####static
#for parameter_id in 0 #1 2 3 4
#do
#  for quant_mode in qint8_mixed_float16 force_float32 force_float16
#  do
#    for batch in 4
#    do 
#      cd $PROJ_ROOT_PATH/export_model
#      bash run.sh 1 $batch
#      cd $PROJ_ROOT_PATH/gen_model
#      bash run.sh 1 $quant_mode false $batch
#      for thread in 1  
#      do
#        MM_RUN $quant_mode false $batch $thread
#      done
#    done
#  done
#done
