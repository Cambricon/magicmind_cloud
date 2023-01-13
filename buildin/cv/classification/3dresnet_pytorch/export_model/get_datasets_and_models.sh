
#!/bin/bash
set -e
set -x

cd $DATASETS_PATH

if [ ! -d "kinetics_videos" ];
then
  echo "Download and preprocess kinetic-dataset for 3dresnet from https://github.com/cvdfoundation/kinetics-dataset"
  exit 1
else
  echo "kinetics_videos already exists."
fi

cd $MODEL_PATH
if [ -f "r3d50_K_200ep.pth" ];
then
  echo "r3d50_K_200ep.pth already exists."
else
  echo "Download r3d50_K_200ep.pth file from https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M"
  exit 1
fi
