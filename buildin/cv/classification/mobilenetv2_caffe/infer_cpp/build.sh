set -e 
set -x
#g++ -std=c++11 -O2 `pkg-config opencv --cflags` -I./include \
#    -I$NEUWARE_HOME/include src/*.cpp -o $PROJ_ROOT_PATH/infer_cpp/infer \
#    -L$NEUWARE_HOME/lib64 -lmagicmind_runtime -lcnrt \
#    -lglog -lgflags -L`pkg-config opencv --variable=libdir` `pkg-config opencv --libs` \
#    -lpthread -Wl,-rpath=`pkg-config opencv --variable=libdir`
g++ -std=c++11 -O2 -Werror `pkg-config opencv --cflags` \
    ${PROJ_ROOT_PATH}/infer_cpp/src/*.cpp ${CPP_COMMON_PATH}/*.cc ${CPP_COMMON_PATH}/*.cpp\
    -I ${NEUWARE_HOME}/include \
    -I ${PROJ_ROOT_PATH}/infer_cpp/include \
    -I ${CPP_COMMON_PATH} \
    -L ${NEUWARE_HOME}/lib64 \
    -o ${PROJ_ROOT_PATH}/infer_cpp/infer \
    -lmagicmind_runtime -lcnrt -lcndrv -lgflags `pkg-config opencv --libs`

