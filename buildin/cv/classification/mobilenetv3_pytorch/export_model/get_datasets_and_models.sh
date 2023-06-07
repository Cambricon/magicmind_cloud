#bin/bash
set -e
set -x

ORIGIN_PT='mobilenetv3_small_67.4.pth.tar'
if [ ! -d ${MODEL_PATH} ];then
    mkdir -p ${MODEL_PATH}
fi
cd ${MODEL_PATH}
if [ ! -f ${ORIGIN_PT} ];then
  echo "mobilenetv3_small_67.4.pth.tar"
  gdown -c https://drive.google.com/uc?id=1lCsN3kWXAu8C30bQrD2JTZ7S2v4yt23C -O ${ORIGIN_PT}
fi

if [ ! -f "pytorch-mobilenet-v3.zip" ];then
  echo "pytorch-mobilenet-v3.zip"
  wget -c https://github.com/kuan-wang/pytorch-mobilenet-v3/archive/refs/heads/master.zip -O pytorch-mobilenet-v3.zip
  unzip -o pytorch-mobilenet-v3.zip -d ${PROJ_ROOT_PATH}/export_model
fi

if [ ! -d ${ILSVRC2012_DATASETS_PATH} ];then
    mkdir -p ${ILSVRC2012_DATASETS_PATH}
fi

cd ${ILSVRC2012_DATASETS_PATH}
echo "Downloading ILSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/ to ${ILSVRC2012_DATASETS_PATH} "
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download ILSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi

