#!/bin/bash
PRECISION=$1

if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
  mkdir "$PROJ_ROOT_PATH/data/output"
  echo "mkdir sucessed!!!"
else
  echo "output dir exits!!! no need to mkdir again!!!"
fi

MAGICMIND_MODEL=$MODEL_PATH/hoitransformer_${PRECISION}_1.mm
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/
if [ ! -d "$OUTPUT_DIR" ];
then
  mkdir "$OUTPUT_DIR"
fi
echo "infer Magicmind model..."

cd ../export_model/HoiTransformer/data/
if [ ! -d "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/hoia" ];
then
  ln -s "$HOIA_DATASETS_PATH" "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/" 
fi

if [ ! -d "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/hico" ];
then
  ln -s "$HOIA_DATASETS_PATH/hico" "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/"
fi
  
if [ ! -d "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/vcoco" ];
then
  ln -s "$HOIA_DATASETS_PATH/vcoco" "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/"
fi

echo "annotation file exists"

cd hoia
if [ ! -d "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/hoia/images" ];
then
  mkdir "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/hoia/images"
fi
echo "images file exists"
if [ ! -d "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/hoia/images/test" ];
then
  ln -s "$HOIA_DATASETS_PATH/images/test" "$PROJ_ROOT_PATH/export_model/HoiTransformer/data/hoia/images/test"
fi

echo "test file exists"
cd $PROJ_ROOT_PATH/infer_python/
ln -s $PROJ_ROOT_PATH/export_model/HoiTransformer/data/ ./
export PYTHONPATH=$PROJ_ROOT_PATH/export_model/HoiTransformer:$PYTHONPATH
python infer.py --model_path $MAGICMIND_MODEL \
	        --backbone=resnet50 \
                --log_dir $OUTPUT_DIR \
		--batch_size=1 \
		--dataset_file=hoia
