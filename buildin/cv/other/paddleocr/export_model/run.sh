#bin/bash
set -e
set -x
## get model and datasets

function gen_onnx_core(){
	model_dir="${1}"
	model_filename="${2}"
	params="${3}"
	onnx_model_name="${4}"
    paddle2onnx --model_dir ${model_dir} \
     --model_filename ${model_filename} \
     --params ${params} \
	 --opset_version 11 \
     --save_file ${onnx_model_name}
}

function gen_onnx(){
    paddle_model_path="${1}"
	model_name=$(basename "${paddle_model_path}")
	onnx_model_name="${model_name}.onnx"
    if [ ! -f "${onnx_model_name}" ];then
        echo "Generating ${onnx_model_name} ..."
    	gen_onnx_core "${paddle_model_path}" \
    	    "${paddle_model_path}/inference.pdmodel" \
    	    "${paddle_model_path}/inference.pdiparams" \
    	    "${onnx_model_name}"
    else
    	echo "${onnx_model_name} already exists!!! No need to generate."
    fi

}

# get ini model and dataset
source get_datasets_and_models.sh

# convert paddle model to onnx model.
pushd ${MODEL_PATH}
    gen_onnx "${det_infer_model_path}"
    gen_onnx "${rec_infer_model_path}"
    gen_onnx "${cls_infer_model_path}"
popd