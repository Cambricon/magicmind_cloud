set -e
set -x

g++ -std=c++11 -O2 -Werror `pkg-config opencv --cflags` \
    -I ${PROJ_ROOT_PATH}/infer_cpp/include \
    -I ${NEUWARE_HOME}/include \
    -I ${CPP_COMMON_PATH} \
    ${PROJ_ROOT_PATH}/infer_cpp/src/*.cpp ${CPP_COMMON_PATH}/*.cc ${CPP_COMMON_PATH}/*.cpp\
    -L ${NEUWARE_HOME}/lib64 \
    -o ${PROJ_ROOT_PATH}/infer_cpp/infer \
    -lmagicmind_runtime -lcnrt -lcndrv -lgflags `pkg-config opencv --libs`

