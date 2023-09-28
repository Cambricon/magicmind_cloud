
#!/bin/bash
set -e
set -x

cd $HOIA_DATASETS_PATH
cd ..
if [ ! -d "hoia" ];
then
  echo "Downloading hoia.zip"
  gdown 'https://drive.google.com/uc?id=1OO7fE0N71pVxgUW7aOp7gdO5dDTmkr_v'
  unzip -o hoia.zip
else
  echo "hoia annotation already exists."
fi

cd hoia
if [ ! -d "images" ];
then
  echo "make directory images"
  mkdir images
else
  echo "images already exists."
fi

if [ ! -d "hico" ];
then
  echo "Downloading hico.zip"
  gdown 'https://drive.google.com/uc?id=1BanIpXb8UH-VsA9yg4H9qlDSR0fJkBtW'
  unzip hico.zip
else
  echo "hico annotation already exists."
fi

if [ ! -d "vcoco" ];
then
  echo "Downloading vcoco.zip"
  gdown 'https://drive.google.com/uc?id=1vWVScXPsu0KVMtXW8QdLjb25NGLzEPhN'
  unzip vcoco.zip
else
  echo "vcoco annotation already exists."
fi

cd images
if [ ! -d "test" ];
then
  echo "Downloading test_2019.zip"
  gdown 'https://drive.google.com/uc?id=1TXxyK0bQI7y1r-zF_md43K78PjB74Kd7'
  unzip -o test_2019.zip
  mv test_2019 test
else
  echo "test already exists."
fi

cd $MODEL_PATH
if [ -f "res50_hoia_a4caffe.pth" ];
then
  echo "res50_hoia_a4caffe.pth already exists."
else
  echo "Downloading res50_hoia_a4caffe.pth file"
  gdown 'https://drive.google.com/uc?id=1bNrFQ6a8aKBzwWc0MAdG2f24StMP9lhY'
fi
