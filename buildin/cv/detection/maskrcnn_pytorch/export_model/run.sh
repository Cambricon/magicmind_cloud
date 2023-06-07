#bin/bash
set -e
set -x
# get model
bash get_datasets_and_models.sh
# convert model
cd $PROJ_ROOT_PATH/export_model
if [ ! -f $MODEL_PATH/maskrcnn.onnx ];then
    python mmdetection/tools/deployment/pytorch2onnx.py --output-file $MODEL_PATH/maskrcnn.onnx mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py $MODEL_PATH/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
                                                        --dynamic-export 
else
    echo "maskrcnn.onnx exist!"
fi
