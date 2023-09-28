#bin/bash
set -e 
set -x

cd $PROJ_ROOT_PATH/export_model/
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
  for shape_mutable in false
  do
    for batch in 1
    do
      MM_MODEL="psenet_tf_${precision}_${shape_mutable}_${batch}"
      if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];
      then
          cd $PROJ_ROOT_PATH/gen_model/
          bash run.sh $precision $shape_mutable $batch 
      fi
      #infer python
      cd $PROJ_ROOT_PATH/infer_python/
      bash run.sh $precision $shape_mutable 500
      #compute hmean
      THIS_OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/psenet_tf_result_${precision}_${shape_mutable}_${batch}.json
      python $UTILS_PATH/compute_icdar_hmean.py  --label_file  $ICDAR_DATASETS_PATH/icdar2015/icdar2015_test_label.json \
                                                 --result_dir $THIS_OUTPUT_DIR  \
                                                 2>&1 | tee $PROJ_ROOT_PATH/data/output/psenet_tf_${precision}_${shape_mutable}_${batch}_log_eval
      
    done
  done
done
