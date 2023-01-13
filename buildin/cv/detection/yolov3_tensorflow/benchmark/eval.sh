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
      MM_MODEL="yolov3_tf_${precision}_${shape_mutable}_${batch}"
      if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];
      then
          cd $PROJ_ROOT_PATH/gen_model/
          bash run.sh $precision $shape_mutable $batch 
      fi
      #infer cpp
      cd $PROJ_ROOT_PATH/infer_cpp/
      bash run.sh $precision $shape_mutable 5000
      #compute coco
      THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/infer_cpp_output_${precision}_${shape_mutable}_${batch}"
      python $UTILS_PATH/compute_coco_mAP.py  --file_list  $UTILS_PATH/coco_file_list_5000.txt \
                                              --result_dir $THIS_OUTPUT_DIR/  \
                                              --ann_dir $DATASETS_PATH/ \
                                              --data_type val2017 \
                                              --json_name $PROJ_ROOT_PATH/data/output/yolov3_tf_${precision}_${shape_mutable}_${batch} \
                                              --img_dir $DATASETS_PATH/val2017 \
                                              --image_num 5000 2>&1 | tee $PROJ_ROOT_PATH/data/output/yolov3_tf_${precision}_${shape_mutable}_${batch}_log_eval
      
    done
  done
done
