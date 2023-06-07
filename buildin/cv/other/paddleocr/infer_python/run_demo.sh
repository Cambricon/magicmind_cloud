set -x
set -e

magicmind_det_model=${1}
magicmind_rec_model=${2}
magicmind_cls_model=${3}

# run det demo on some image
# cd ${PROJ_ROOT_PATH}/export_model/PaddleOCR
# python tools/infer/predict_det.py --image_dir ./doc/imgs \
#                                   --det_model_dir  ${magicmind_det_model} \
#                                   --draw_img_save_dir ./det_infer_results

# run rec demo on some words images
# cd ${PROJ_ROOT_PATH}/export_model/PaddleOCR
# python tools/infer/predict_rec.py --image_dir ./doc/imgs_words/ch \
#                                   --rec_model_dir  ${magicmind_rec_model}  \
#                                   --draw_img_save_dir ./rec_infer_results

# run rec demo on some words images
# cd ${PROJ_ROOT_PATH}/export_model/PaddleOCR
# python tools/infer/predict_cls.py --image_dir ./doc/imgs_words/ch \
#                                   --cls_model_dir  ${magicmind_cls_model}  \
#                                   --draw_img_save_dir ./rec_infer_results

# run det+cls+rec e2e demo, you could choose use cls or not.
cd ${PROJ_ROOT_PATH}/export_model/PaddleOCR
python tools/infer/predict_system.py --image_dir ./doc/imgs \
                                     --det_model_dir ${magicmind_det_model} \
                                     --rec_model_dir  ${magicmind_rec_model}  \
                                     --use_angle_cls True \
                                     --cls_model_dir  ${magicmind_cls_model}  \
                                     --draw_img_save_dir ./system_infer_results



