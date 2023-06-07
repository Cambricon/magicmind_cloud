#!/bin/bash
set -e
set -x

backbone="${1:-COCO}"
if [ ${backbone} != "COCO" ]  && [ ${backbone} != "BODY_25" ];then
    echo "backbone MUST BE COCO or BODY_25, now is ${backbone}!"
    exit 1
fi

if [ -d ${MODEL_PATH} ];
then
    echo "folder ${MODEL_PATH} already exist!!!"
else
    mkdir -p "${MODEL_PATH}"
fi

cd ${COCO_DATASETS_PATH}

if [ ! -d "val2017" ];then 
    echo "Downloading val2017.zip"
    wget -c http://images.cocodataset.org/zips/val2017.zip
    unzip -o val2017.zip
else 
    echo "val2017 already exists."
fi

if [ ! -d "annotations" ];then
    echo "Downloading annotations_trainval2017.zip"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -o annotations_trainval2017.zip
else 
    echo "annotations_trainval2017 already exists."
fi

cd ${MODEL_PATH}
caffemodel=openpose.caffemodel
prototxt=openpose.prototxt

if [ ${backbone} == "COCO" ];then

    if [ -f "pose_iter_440000.caffemodel" ]; then
        echo "pose_iter_440000.caffemodel already exists."
    else
        echo "Downloading pose_iter_440000.caffemodel file"
        wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
    fi
    
    if [ -f "pose_deploy_linevec.prototxt" ]; then
        echo "pose_deploy_linevec.prototxt already exists."
    else
        echo "Downloading pose_deploy_linevec.prototxt file"
        wget -c https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
    fi


else
# BODY_25 
    if [ -f "pose_iter_584000.caffemodel" ]; then
        echo "pose_iter_584000.caffemodel already exists."
    else
        echo "Downloading pose_iter_584000.caffemodel file"
        wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel
    fi

    if [ -f "pose_deploy.prototxt" ]; then
        echo "pose_deploy.prototxt already exists."
    else
        echo "Downloading pose_deploy.prototxt file"
        wget -c https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt
    fi


fi

# update soft-link
if [ -f ${caffemodel} ];then
    rm ${caffemodel}
fi
if [ -f ${prototxt} ];then
    rm ${prototxt}
fi

if [ ${backbone} == "COCO" ];then
    ln -s pose_iter_440000.caffemodel ${caffemodel}
    ln -s pose_deploy_linevec.prototxt ${prototxt}
else
    ln -s pose_iter_584000.caffemodel ${caffemodel}
	ln -s pose_deploy.prototxt ${prototxt}
fi
