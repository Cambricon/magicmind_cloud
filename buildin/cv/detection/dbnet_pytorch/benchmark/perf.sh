#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    N=$3
    H=$4
    W=$5
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    if [ ${SHAPE_MUTABLE} == 'false' ];
    then
        model_path=$MODEL_PATH/dbnet_pt_model_${PRECISION}_${SHAPE_MUTABLE}_${N}_${H}_${W}
    else 
        model_path=$MODEL_PATH/dbnet_pt_model_${PRECISION}_${SHAPE_MUTABLE}
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $model_path \
                          --iterations 1000 \
                          --input_dims ${N},3,${H},${W} \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${N}_${H}_${W}_log_perf
}

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
  for shape_mutable in false #true
  do
    for n in 1 16 32
    do
      for h in 736
      do
        for w in 1280
        do
          cd $PROJ_ROOT_PATH/gen_model
          bash run.sh $precision $shape_mutable $n $h $w
          MM_RUN $precision $shape_mutable $n $h $w
        done
      done
    done
  done
done
