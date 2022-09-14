#!/bin/bash
set -e
set -x

cd $MODEL_PATH
if [ -f "Task002_Heart.zip" ];
then
  echo "Task002_Heart.zip already exists."
else
  echo "Downloading models ..."
  wget -c https://zenodo.org/record/4003545/files/Task002_Heart.zip?download=1 -O Task002_Heart.zip
  unzip Task002_Heart.zip
fi

cd $nnUNet_raw_data_base
if [ -d "Task02_Heart" ];
then
  echo "Task02_Heart already exist."
else
  echo "Please download Task02_Heart.tar on https://drive.google.com/uc?id=1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY to $nnUNet_raw_data_base and tar -xf Task02_Heart.tar."
fi
