#bin/bash
set -e
set -x

if [ ! -d $DATASETS_PATH ];
then
  mkdir -p $DATASETS_PATH
fi 


if [ ! -d $MODEL_PATH ];
then
  mkdir -p $MODEL_PATH
fi 

if [ ! -d "FSA-Net" ];
then
    git clone https://github.com/shamangary/FSA-Net
    cd FSA-Net
    git reset --hard 4361d0e48103bb215d15734220c9d17e6812bb4
else
    echo "FSA_Net already exist !"
fi

cd $DATASETS_PATH
cp $PROJ_ROOT_PATH/export_model/FSA-Net/data/type1/AFLW2000.npz .

cd $MODEL_PATH
cp $PROJ_ROOT_PATH/export_model/FSA-Net/pre-trained/converted-models/tf/* .

echo "datasets and models prepare OK !"

