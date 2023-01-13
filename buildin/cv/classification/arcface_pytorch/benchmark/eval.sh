#!/bin/bash
set -e
set -x

PRECISION=$1
BATCH_SIZE=$2

COMPUTE_ACCURACY(){
    PRECISION=$1
    BATCH_SIZE=$2
    python ${UTILS_PATH}/ijbc_eval.py --features_dir ${PROJ_ROOT_PATH}/data/images/${PRECISION}_${BATCH_SIZE} \
	                              --output_file $PROJ_ROOT_PATH/data/output/${PRECISION}_${BATCH_SIZE} \
				      --face_tid_mid_file $DATASETS_PATH/IJBC/meta/ijbc_face_tid_mid.txt \
                                      --template_pair_label_file $DATASETS_PATH/IJBC/meta/ijbc_template_pair_label.txt
}

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi

echo "Parm doesn't exist, run benchmark"
cd $PROJ_ROOT_PATH/export_model
bash run.sh
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision 64	
    for batch in 64
    do
        cd $PROJ_ROOT_PATH/infer_cpp
        bash run.sh $precision $batch 469375
        COMPUTE_ACCURACY $precision $batch
    done
done
