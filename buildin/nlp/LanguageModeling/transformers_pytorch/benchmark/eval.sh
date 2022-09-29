#bin/bash
set -e 
set -x

cd $PROJ_ROOT_PATH/export_model/
bash run.sh

#static cpp
DEV_ID=0

for quant_mode in force_float32 force_float16
do
  for shape_mutable in false
  do
    for batch in 1 4 8
    do
      MM_MODEL="${quant_mode}_${shape_mutable}_${batch}"
      if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
          cd $PROJ_ROOT_PATH/gen_model/
          bash run.sh $quant_mode $shape_mutable $batch
      fi
      #infer python
      cd $PROJ_ROOT_PATH/infer_python/
      bash run.sh $quant_mode $shape_mutable $batch
  
      #compare_eval
      python $MAGICMIND_CLOUD/test/compare_eval.py  --metric mrpc \
                                                    --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}_acc_and_f1_result.txt \
                                                    --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${shape_mutable}_${batch}_acc_and_f1_result.txt \
                                                    --model transformers
    done
  done
done
