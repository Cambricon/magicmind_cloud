#!/bin/bash
set -e
set -x

MM_RUN(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    THREADS=$4
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                          --iterations 1000 \
                          --batch ${BATCH_SIZE} \
                          --threads ${THREADS} \
                          --devices 0 2>&1 | tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf
}

#static
cd $PROJ_ROOT_PATH/export_model/
bash run.sh
cd $PROJ_ROOT_PATH/gen_model/
mkdir -p "$PROJ_ROOT_PATH/data/output/"
for quant_mode in force_float32 force_float16 qint8_mixed_float16
do
  for batch in 1 4 8
  do
    for shape_mutable in false
    do
        MM_MODEL="${quant_mode}_false_${batch}"
        if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
            bash run.sh $quant_mode $shape_mutable $batch
        fi
        for threads in 1  
        do
          MM_RUN $quant_mode $shape_mutable $batch $threads
          # compare perf
          python $MAGICMIND_CLOUD/test/compare_perf.py  --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}_log_perf \
                                                        --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${shape_mutable}_${batch}_log_perf \
                                                        --model c3d_caffe
        done
      done
  done
done
