#bin/bash
#Xiao Qi 2022-7-29
set -e
set -x

FILE1="UCF101"
FILE2="UCF101TrainTestSplits-RecognitionTask.zip"
FILE3="c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel"
FILE4="c3d_resnet18_r2_ucf101.prototxt"

if [ ! -d "${UCF101_DATASETS_PATH}" ];then
    mkdir -p ${UCF101_DATASETS_PATH}
    cd ${UCF101_DATASETS_PATH}/../
    wget -c https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate -O UCF101.rar
    unrar x -o+ UCF101.rar
else
echo "${FILE1} Exist!"
fi 

cd ${UCF101_DATASETS_PATH}
if [ ! -f "${UCF101_DATASETS_PATH}/${FILE2}" ];then
    wget -c https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate -O ${UCF101_DATASETS_PATH}/UCF101TrainTestSplits-RecognitionTask.zip
else
echo "${FILE2} Exist!"
fi

if [ ! -d "${UCF101_DATASETS_PATH}/ucfTrainTestlist" ];then
    apt-get update
    apt-get install dos2unix -y
    unzip -o "UCF101TrainTestSplits-RecognitionTask.zip"
    find ucfTrainTestlist/ -name "*.txt" | xargs dos2unix
else
    echo "ucfTrainTestlist Exist!"
fi

if [ ! -d ${PROJ_ROOT_PATH}/data/models ];then
    mkdir -p ${PROJ_ROOT_PATH}/data/models
fi

if [ ! -f "${PROJ_ROOT_PATH}/data/models/${FILE3}" ];then
    wget -c https://www.dropbox.com/s/bf5z2jw1pg07c9n/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0 -O ${PROJ_ROOT_PATH}/data/models/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel
else
    echo "${FILE3} Exist!"
fi

if [ ! -f "${PROJ_ROOT_PATH}/data/models/${FILE4}" ];then
    wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/c3d_resnet18_r2_ucf101.prototxt -O ${PROJ_ROOT_PATH}/data/models/c3d_resnet18_r2_ucf101.prototxt
else
    echo "${FILE4} Exist!"
fi
echo "Finish!"
