#bin/bash
set -e 
set -x

cd $PROJ_ROOT_PATH/export_model/
bash run.sh

#static cpp
DEV_ID=0

for precision in force_float32 force_float16
do
  for shape_mutable in true
  do
    for batch in 1 16 32
    do
      MM_MODEL="${precision}_${shape_mutable}_${batch}"
      if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
          cd $PROJ_ROOT_PATH/gen_model/
          bash run.sh $precision $shape_mutable $batch
      fi
      #infer python
      cd $PROJ_ROOT_PATH/infer_python/
      bash run.sh $precision $shape_mutable $batch
    done
  done
done
