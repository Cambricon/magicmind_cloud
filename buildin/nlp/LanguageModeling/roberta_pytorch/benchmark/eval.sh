set -x

#dynamic
for max_seq_length in 128
do 
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh 1 $max_seq_length
    for precision in force_float32 force_float16
    do 
        cd $PROJ_ROOT_PATH/gen_model
	bash run.sh $precision true 1 $max_seq_length
	for batch in 32
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $precision true $batch $max_seq_length 
        done
    done
done
