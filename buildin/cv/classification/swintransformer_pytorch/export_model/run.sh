#!/bin/bash
set -e
set -x
bash get_datasets_and_models.sh

if [ -f "$MODEL_PATH/swin.onnx" ];
then
  echo "swin.onnx already exists."
else
  echo "exporting swin.onnx"
  python export_onnx.py
fi

