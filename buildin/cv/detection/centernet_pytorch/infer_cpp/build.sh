g++ -std=c++11 -O2 `pkg-config opencv --cflags` \
    -I $PROJ_ROOT_PATH/infer_cpp \
    -I $NEUWARE_HOME/include $PROJ_ROOT_PATH/infer_cpp/src/*.cpp -o $PROJ_ROOT_PATH/infer_cpp/infer \
    -L$NEUWARE_HOME/lib64 -lmagicmind_runtime -lcnrt \
    -lglog -lgflags `pkg-config opencv --libs`

