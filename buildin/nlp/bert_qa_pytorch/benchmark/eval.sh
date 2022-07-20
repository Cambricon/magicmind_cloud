#!/bin/bash
set -e
set -x

#dynamic
for max_seq_length in 128
do 
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh 1 $max_seq_length
    for quant_mode in force_float32 force_float16
    do 
        cd $PROJ_ROOT_PATH/gen_model
	bash run.sh $quant_mode true 1 $max_seq_length
	for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $quant_mode true 1 $batch $max_seq_length 
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric squad --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_${quant_mode}_true_${batch}bs_${max_seq_length}/acc_result.txt --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_python_output_${quant_mode}_true_${batch}bs_${max_seq_length}_acc_result.txt --model bert_qa_pytorch
        done
    done
done
