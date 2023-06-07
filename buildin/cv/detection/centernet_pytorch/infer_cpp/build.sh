g++ -std=c++11 -O2 `pkg-config opencv --cflags` \
    $PROJ_ROOT_PATH/infer_cpp/src/*.cpp $CPP_COMMON_PATH/*.cc \
    -I $NEUWARE_HOME/include \
    -I $PROJ_ROOT_PATH/infer_cpp/include \
    -I $CPP_COMMON_PATH \
    -L $NEUWARE_HOME/lib64 \
    -o $PROJ_ROOT_PATH/infer_cpp/infer \
    -lmagicmind_runtime -lcnrt -lcndrv -lgflags `pkg-config opencv --libs`

