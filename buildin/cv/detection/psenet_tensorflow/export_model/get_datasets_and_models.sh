#bin/bash
set -e
set -x

if [ ! -d $ICDAR_DATASETS_PATH ];
then
  mkdir -p $ICDAR_DATASETS_PATH
fi 
cd $ICDAR_DATASETS_PATH

DATA="icdar2015"
if [ ! -d $DATA ];
then
  echo "Downloading icdar2015.tar.gz"
  gdown -c https://drive.google.com/uc?id=1msjWJ00sO90GhqSNkkGECZzqV9poA1NA
  tar -xvf icdar2015.tar.gz
fi

if [ ! -d $MODEL_PATH ];
then
  mkdir -p $MODEL_PATH
fi 
cd $MODEL_PATH

if [ -f  model.tar.gz ];
then
    tar -xvf model.tar.gz
else
    gdown -c https://drive.google.com/uc?id=1TjJvtwMp8hJXQhn6Yz2lbPdvBGH-ZQ8u
    tar -xvf model.tar.gz
fi
