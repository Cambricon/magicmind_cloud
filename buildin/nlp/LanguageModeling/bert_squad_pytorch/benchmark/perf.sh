#!/bin/bash
set -e
set -x

MM_RUN(){
    magicmind_model=$1
    batch_size=$2
    max_length=$3
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi

    ${MM_RUN_PATH}/mm_run   --magicmind_model ${magicmind_model} \
                            --batch_size ${batch_size} ${batch_size} ${batch_size} \
                            --input_files $PROJ_ROOT_PATH/data/input0_${batch_size}_${max_length}.bin \
                                          $PROJ_ROOT_PATH/data/input1_${batch_size}_${max_length}.bin \
                                          $PROJ_ROOT_PATH/data/input2_${batch_size}_${max_length}.bin \
                            --devices 0
}


dynamic_shape=false
max_seq_length=384
# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh $max_seq_length


for precision in force_float32 force_float16
do 
    for dynamic_shape in false 
    do 
        for batch_size in 1 16 32
        do 
            magicmind_model=${MODEL_PATH}/bert_squad_pytorch_model_${precision}_${dynamic_shape}
            if [ ${dynamic_shape} == 'false' ];then
                magicmind_model="${magicmind_model}_${batch_size}"
            fi

            # gen model
            if [ ! -f ${magicmind_model} ];then
                cd ${PROJ_ROOT_PATH}/gen_model
                bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
            else
                echo "MagicMind model: ${magicmind_model} already exists!"
            fi
	    # run model
	    MM_RUN ${magicmind_model} ${batch_size} ${max_seq_length}
        done 
    done 
done


