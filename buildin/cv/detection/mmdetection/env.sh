### 在开始运行本仓库前先检查以下路径：
#数据集路径
#export COCO_DATASETS_PATH=/path/to/modelzoo/datasets/coco
if [ -z ${COCO_DATASETS_PATH} ] || [ ! -d ${COCO_DATASETS_PATH} ];then
    echo "Error: COCO_DATASETS_PATH is not found, please set it and export it to env!"
fi

export NEUWARE_HOME=/usr/local/neuware
export MM_RUN_PATH=${NEUWARE_HOME}/bin
#本sample工作路径
export PROJ_ROOT_PATH=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
export MAGICMIND_CLOUD=${PROJ_ROOT_PATH%buildin*}
export MODEL_PATH=${PROJ_ROOT_PATH}/data/models

# CV类网络通用文件路径
export UTILS_PATH=${MAGICMIND_CLOUD}/buildin/cv/utils

# Python公共组件路径
export PYTHON_COMMON_PATH=${MAGICMIND_CLOUD}/buildin/python_common
# CPP公共接口路径
export CPP_COMMON_PATH=$MAGICMIND_CLOUD/buildin/cpp_common

has_add_common_path=$(echo ${PYTHONPATH}|grep "${PYTHON_COMMON_PATH}")
if [ -z ${has_add_common_path} ];then
    export PYTHONPATH=${PYTHONPATH}:${PYTHON_COMMON_PATH}
fi

has_add_util_path=$(echo ${PYTHONPATH}|grep "${UTILS_PATH}")
if [ -z ${has_add_util_path} ];then
    export PYTHONPATH=${PYTHONPATH}:${UTILS_PATH}
fi

declare -A config_paths 
declare -A pretrained_paths
declare -A model_image_sizes 
config_paths=( 
    [Mask_R-CNN]="${PROJ_ROOT_PATH}/export_model/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py" \
    [Faster_R-CNN]="${PROJ_ROOT_PATH}/export_model/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py" \
    [RetinaNet]="${PROJ_ROOT_PATH}/export_model/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py" \
    [SSD]="${PROJ_ROOT_PATH}/export_model/mmdetection/configs/ssd/ssd512_coco.py" \
    [Cascade_R-CNN]="${PROJ_ROOT_PATH}/export_model/mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco.py" \
    [HRNet]="${PROJ_ROOT_PATH}/export_model/mmdetection/configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py"
)
pretrained_paths=( 
    [Mask_R-CNN]="https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth" \
    [Faster_R-CNN]="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth" \
    [RetinaNet]="https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth" \
    [SSD]="https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth" \
    [Cascade_R-CNN]="https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco/cascade_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.423_20200504_175649-cab8dbd5.pth" \
    [HRNet]="https://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth"
)
model_image_sizes=( 
    [Mask_R-CNN]="800,800" \
    [Faster_R-CNN]="800,800" \
    [RetinaNet]="800,800" \
    [SSD]="512,512" \
    [Cascade_R-CNN]='800,800',
    [HRNet]='800,800'
)

# 用户可自行选择其中一个模型，项目将基于用户所选模型进行MagicMind 模型构建，模型推理测试以及模型性能测试
# 索引从0起始 指定完成之后 务必执行source env.sh 使该环境变量生效
MMDETECTION_MODEL_NAMES=(
    Faster_R-CNN \
    Mask_R-CNN \
    RetinaNet \
    SSD \
    Cascade_R-CNN \
    HRNet
)
# choose which model to build and infer index begin from 0
export MMDETECTION_MODEL_NAME=${MMDETECTION_MODEL_NAMES[3]}
# choose model image size
export MMDETECTION_MODEL_IMAGE_SIZE=${model_image_sizes[${MMDETECTION_MODEL_NAME}]}
# choose config file path
export MMDETECTION_MODEL_CONFIG_PATH=${config_paths[${MMDETECTION_MODEL_NAME}]}
# choose pretrained file path
export MMDETECTION_MODEL_PRETRAINED_PATH=${pretrained_paths[${MMDETECTION_MODEL_NAME}]}

echo "MMDetection Model: ${MMDETECTION_MODEL_NAME}"
echo "MMDetection Model Config URL: ${MMDETECTION_MODEL_CONFIG_PATH}"
echo "MMDetection Model Pretrained URL: ${MMDETECTION_MODEL_PRETRAINED_PATH}"
echo "MMDetection Model Image Size: ${MMDETECTION_MODEL_IMAGE_SIZE}"
