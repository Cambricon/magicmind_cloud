#bin/bash
set -e 
set -x

cd $PROJ_ROOT_PATH/export_model/
bash run.sh
#static cpp
for percision in force_float32 force_float16 qint8_mixed_float16
do
  for shape_mutable in true
  do
    for batch in 1
    do
      #shape_mutable为true时 生成的模型与batch_size无关 故batch_size恒定设置为1即可
      if [ $shape_mutable == "true" ];then
        MM_MODEL="${percision}_${shape_mutable}_1"
        GEN_BATCH=1
      else
        MM_MODEL="${percision}_${shape_mutable}_${batch}"
        GEN_BATCH=$batch
      fi 
      if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
          cd $PROJ_ROOT_PATH/gen_model/
          bash run.sh $percision $shape_mutable $GEN_BATCH
      fi
      #infer cpp
      cd $PROJ_ROOT_PATH/infer_cpp/
      bash run.sh $percision $shape_mutable $batch
      #compute coco
      THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/${percision}_${shape_mutable}_${batch}"
      python $UTILS_PATH/compute_coco_mAP.py  --file_list ${THIS_OUTPUT_DIR}/json/image_name.txt \
                                              --result_dir $THIS_OUTPUT_DIR/results \
                                              --ann_dir $DATASETS_PATH/ \
                                              --data_type 'val2017' \
                                              --json_name $THIS_OUTPUT_DIR/json/${percision}_${shape_mutable}_${batch} \
                                              --img_dir $DATASETS_PATH/val2017 \
                                              --image_num 5000 2>&1 | tee $PROJ_ROOT_PATH/data/output/${percision}_${shape_mutable}_${batch}_log_eval
    done
  done
done
