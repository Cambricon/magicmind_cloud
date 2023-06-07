#!/bin/bash
set -e
set -x

MM_RUN(){
    MAGICMIND_MODEL=${1}
    BATCH_SIZE=${2}
    MAX_SEQ_LENGTH=${3}
    if [ ! -d ${PROJ_ROOT_PATH}/data/output ];
    then
        mkdir "${PROJ_ROOT_PATH}/data/output"
    fi

    ${MM_RUN_PATH}/mm_run --magicmind_model ${MAGICMIND_MODEL} \
                          --iterations 1000 \
                          --input_dims ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} \
                          --devices 0 \
			              --input_files ${PROJ_ROOT_PATH}/data/input0_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin ${PROJ_ROOT_PATH}/data/input1_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin ${PROJ_ROOT_PATH}/data/input2_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin 
}


for max_seq_length in 128
do
    for batch in 1 16 32
    do
        cd $PROJ_ROOT_PATH/export_model
        onnx_path=${MODEL_PATH}/roberta_${batch}bs_128.onnx
        bash run.sh $batch $max_seq_length ${onnx_path}
        for precision in force_float32 force_float16
        do
           magicmind_model=${MODEL_PATH}/roberta_${precision}_false_${batch}_128
           cd $PROJ_ROOT_PATH/gen_model
           bash run.sh ${magicmind_model} ${precision} ${batch} false ${max_seq_length} ${onnx_path} 
           MM_RUN ${magicmind_model} ${batch} ${max_seq_length}
	done
    done
done
