#bin/bash
set -e 
set -x

cd $PROJ_ROOT_PATH/export_model/
bash run.sh

#dynamic and bs>8 will report error:device has no memory to alloc
for precision in force_float32
do
  for shape_mutable in true
  do
    for batch in 1 
    do
      #shape_mutable为true时 生成的模型与batch_size无关 故batch_size恒定设置为1即可
      if [ $shape_mutable == "true" ];then
        MM_MODEL="${precision}_${shape_mutable}_1"
        GEN_BATCH=1
      else
        MM_MODEL="${precision}_${shape_mutable}_${batch}"
        GEN_BATCH=$batch
      fi 
      if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
          cd $PROJ_ROOT_PATH/gen_model/
          bash run.sh $precision $shape_mutable $GEN_BATCH
      fi
      # infer python and compute coco
      cd $PROJ_ROOT_PATH/infer_python/
      bash run.sh $precision $shape_mutable 2>&1 | tee $PROJ_ROOT_PATH/data/output/${precision}_${shape_mutable}_${batch}_log_eval
    done
  done
done
