#!/bin/bash
set -e
set -x

#dynamic
echo "Parm doesn't exist, run benchmark"
for parameter_id in 0 #1 2 3 4
do
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh $parameter_id 1
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $parameter_id $quant_mode true 1
	for batch in 1
	do 
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $parameter_id $quant_mode true 1 $batch
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric unet --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_${quant_mode}_true_${batch}bs_${parameter_id}/summary.json --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_python_output_${quant_mode}_true_${batch}bs_${parameter_id}_summary.json --model nnUNet_pytorch
        done
    done
done
##static
#for parameter_id in 0 #1 2 3 4
#do
#  for quant_mode in force_float32 force_float16 qint8_mixed_float16
#  do
#    for batch in 4
#    do
#      cd $PROJ_ROOT_PATH/export_model
#      bash run.sh $parameter_id $batch
#      cd $PROJ_ROOT_PATH/gen_model
#      bash run.sh $parameter_id $quant_mode false $batch
#      cd $PROJ_ROOT_PATH/infer_python
#      bash run.sh $parameter_id $quant_mode true $batch
#    done
#  done
#done
