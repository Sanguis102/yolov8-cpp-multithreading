#!/bin/bash
set -e

TARGET_SOC="rk3588"
GCC_COMPILER=aarch64-linux-gnu

export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../..
make -j$(nproc)
make install
cd -

# relu版本
cd install/rknn_yolov8_demo_Linux/ && ./rknn_yolov8_demo ./model/RK3588/yolov8s-640-640.rknn ../../720p60hz.mp4
# 使用摄像头
# cd install/rknn_yolov8_demo_Linux/ && ./rknn_yolov8_demo ./model/RK3588/yolov8s-640-640.rknn 0

