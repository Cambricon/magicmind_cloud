#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/mm_model/psenet_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                          --iterations 1000 \
                          --batch_size ${BATCH_SIZE} \
                          --devices 0 2>&1 | tee $PROJ_ROOT_PATH/data/output/psenet_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf
}

#static
cd $PROJ_ROOT_PATH/export_model/
bash run.sh
cd $PROJ_ROOT_PATH/gen_model/
if [ ! -d $PROJ_ROOT_PATH/data/output/ ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output/
fi 

for precision in force_float32 force_float16 qint8_mixed_float16
do
    for batch in 1 16 32
    do
        for shape_mutable in false
        do
            MM_MODEL="psenet_tf_${precision}_${shape_mutable}_${batch}"
            if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];
            then
                bash run.sh $precision $shape_mutable $batch
            fi
            # mm run
            MM_RUN $precision $shape_mutable $batch
        done
    done
done
