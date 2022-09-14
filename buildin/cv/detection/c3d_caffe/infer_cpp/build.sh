set -e 
set -x
g++ -std=c++11 -O2 `pkg-config opencv --cflags` -I./include \
    -I$NEUWARE_HOME/include src/*.cpp -o $PROJ_ROOT_PATH/infer_cpp/infer \
    -L$NEUWARE_HOME/lib64 -lmagicmind_runtime -lcnrt \
    -lglog -lgflags -L`pkg-config opencv --variable=libdir` `pkg-config opencv --libs` \
    -lpthread -Wl,-rpath=`pkg-config opencv --variable=libdir`
