#!/bin/bash
set -e
set -x

#1. get datasets and models
bash get_datasets_and_models.sh

#2. download volo
cd $PROJ_ROOT_PATH/export_model
if [ ! -d volo ]; then git clone https://github.com/sail-sg/volo.git; fi

#3. patch
cd $PROJ_ROOT_PATH/export_model/volo
if grep -q "pt_path" validate.py;
then
  echo "patch has been used"
else
  git apply ../patch
fi

#4. convert model
cd $PROJ_ROOT_PATH/export_model/volo
pt_path=$MODEL_PATH/volo_d2.pt
if [ ! -f $pt_path ]; then
python3 validate.py $DATASETS_PATH  --model volo_d2 \
	            --pt_path $pt_path \
                    --checkpoint $MODEL_PATH/d2_224_85.2.pth.tar --no-test-pool --img-size 224 -b 128
fi
