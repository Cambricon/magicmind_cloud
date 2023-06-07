#!/bin/bash
set -e
set -x

infer_mode=infer_python
image_num=508

cd ${PROJ_ROOT_PATH}/export_model
bash run.sh

for precision in force_float32 force_float16 qint8_mixed_float16
do
    for dynamic_shape in true
    do
	for batch_size in 1
	do
		magicmind_model=${MODEL_PATH}/unet_carvana_model_${precision}_${dynamic_shape}
		if [ ${dynamic_shape} == 'false' ];then
                	magicmind_model="${magicmind_model}_${batch_size}"
            	fi

        	cd ${PROJ_ROOT_PATH}/gen_model
       		bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}
        	cd ${PROJ_ROOT_PATH}/infer_python
        	bash run.sh ${magicmind_model} ${batch_size} 508 1 2>&1 |tee ${PROJ_ROOT_PATH}/data/output/${precision}_${dynamic_shape}_log_eval
    	done
    done	
done

