#!/bin/bash
set -e
set -x

#1. get datasets and models
bash get_datasets_and_models.sh

#2. download Conformer
cd $PROJ_ROOT_PATH/export_model
if [ ! -d Conformer ]; then git clone https://github.com/pengzhiliang/Conformer.git; fi

#3. patch
cd $PROJ_ROOT_PATH/export_model/Conformer
if grep -q "pt_path" engine.py;
then
  echo "patch has been used"
else
  git apply ../patch
fi

#4.convert model
cd $PROJ_ROOT_PATH/export_model/Conformer
pt_path=$MODEL_PATH/conformer_small_patch16.pt
if [ ! -f $pt_path ]; then	
python main.py  --model Conformer_small_patch16 --eval --batch-size 64 \
                --input-size 224 \
		--pt_path $pt_path \
                --data-set IMNET \
                --num_workers 4 \
                --data-path $DATASETS_PATH \
                --epochs 100 \
                --resume $MODEL_PATH/Conformer_small_patch16.pth
fi
