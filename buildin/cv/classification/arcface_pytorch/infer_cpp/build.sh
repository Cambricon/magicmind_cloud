#!/bin/bash
if [ -d  ./bin ];then
  rm -rf ./bin/*
else 
  mkdir bin
fi

g++ -std=c++11 -O2 -Werror `pkg-config opencv --cflags` \
  ${PROJ_ROOT_PATH}/infer_cpp/src/*.cpp ${CPP_COMMON_PATH}/*.cc \
  -I ${PROJ_ROOT_PATH}/infer_cpp/include/ \
  -I ${NEUWARE_HOME}/include  \
  -I ${CPP_COMMON_PATH} \
  -o ./bin/host_infer \
  -L ${NEUWARE_HOME}/lib64 \
  -lmagicmind_runtime -lcnrt -lcndrv  -lgflags `pkg-config opencv --libs`

