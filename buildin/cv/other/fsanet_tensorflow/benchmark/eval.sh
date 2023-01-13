#bin/bash
set -e 
set -x

cd $PROJ_ROOT_PATH/export_model/
bash run.sh
for precision in force_float32 
do
  for shape_mutable in false
  do
    for batch in 1
    do
      MM_MODEL1="fsanet_capsule_${precision}_${shape_mutable}_${batch}"
      if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL1 ];
      then
          cd $PROJ_ROOT_PATH/gen_model/
          bash run.sh $precision $shape_mutable $batch 
      fi
      #infer python and compute MAE
      cd $PROJ_ROOT_PATH/infer_python/
      bash run.sh $precision $shape_mutable 1969      
    done
  done
done
