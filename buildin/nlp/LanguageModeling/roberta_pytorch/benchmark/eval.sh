set -x

#dynamic
for max_seq_length in 128
do 
    onnx_path=${MODEL_PATH}/roberta_${max_seq_length}.onnx
    cd ${PROJ_ROOT_PATH}/export_model
    bash run.sh 1 ${max_seq_length} ${onnx_path}
    for precision in force_float32 force_float16
    do 
        for batch in 32
        do
            magicmind_model=${MODEL_PATH}/roberta_${precision}_false_${batch}_${max_seq_length}
            cd ${PROJ_ROOT_PATH}/gen_model
	        bash run.sh ${magicmind_model} ${precision} ${batch} false ${max_seq_length} ${onnx_path} 
            cd ${PROJ_ROOT_PATH}/infer_python
            bash run.sh ${magicmind_model} ${batch}
        done
    done
done
