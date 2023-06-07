#!/bin/bash
set -e
set -x

infer_mode=infer_python
max_seq_length=384

# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh $max_seq_length

for precision in force_float16 force_float32
do 
    for dynamic_shape in true
    do 
        for batch_size in 4 8
        do 
            magicmind_model=${MODEL_PATH}/bert_squad_pytorch_model_${precision}_${dynamic_shape}
            if [ ${dynamic_shape} == 'false' ];then
                magicmind_model="${magicmind_model}_${batch_size}"
            fi
	    infer_res_path="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_infer_res"

            # gen model
            if [ ! -f ${magicmind_model} ];then
                cd ${PROJ_ROOT_PATH}/gen_model
                bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
            else
                echo "MagicMind model: ${magicmind_model} already exists!"
            fi

            # infer and calc acc
            if [ ${infer_mode} == "infer_python" ];then
                cd ${PROJ_ROOT_PATH}/infer_python
                bash run.sh  ${magicmind_model} ${batch_size} ${max_seq_length}
            fi
        done 
    done 
done




