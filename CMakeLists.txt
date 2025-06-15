cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 89)
#set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
set(PROJECT_NAME "MultiStreamDetector")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Function)
find_package(TBB REQUIRED)                # 寻找已安装的 TBB
# CUDA
find_package(CUDA REQUIRED)
print_var(CUDA_LIBRARIES)
print_var(CUDA_INCLUDE_DIRS)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
print_var(CUDA_LIB_DIR)

# OpenCV
find_package(OpenCV REQUIRED)
print_var(OpenCV_LIBS)
print_var(OpenCV_LIBRARIES)
print_var(OpenCV_INCLUDE_DIRS)

# TensorRT
# 手动指定 TensorRT 安装路径
#set(TensorRT_ROOT /usr/local/TensorRT-8.4.1.5)
#set(TensorRT_INCLUDE_DIR ${TensorRT_ROOT}/include)
#set(TensorRT_LIB_DIR ${TensorRT_ROOT}/lib/libnvinfer.so)
#set(TensorRT_LIBRARY_INFER_PLUGIN ${TensorRT_ROOT}/lib/libnvinfer_plugin.so)
# 直接定义 TensorRT 库，避免使用有问题的 FindTensorRT.cmake
#set(TensorRT_LIBRARIES
#        ${TensorRT_LIB_DIR}/libnvinfer.so
#        ${TensorRT_LIB_DIR}/libnvinfer_plugin.so
#        ${TensorRT_LIB_DIR}/libnvonnxparser.so
#        ${TensorRT_LIB_DIR}/libnvparsers.so
#)

find_package(TensorRT REQUIRED)
print_var(TensorRT_LIBRARIES)
print_var(TensorRT_INCLUDE_DIRS)
print_var(TensorRT_LIB_DIR)
if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
    message(STATUS "Build with -DTRT_10")
    add_definitions(-DTRT_10)
endif ()

list(APPEND ALL_INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
)

list(APPEND ALL_LIB_DIRS
        ${CUDA_LIB_DIR}
        ${TensorRT_LIB_DIR}
)

print_var(ALL_INCLUDE_DIRS)
print_var(ALL_LIBS)
print_var(ALL_LIB_DIRS)

include_directories(#包含头文件目录
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(
        ${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${ALL_INCLUDE_DIRS}
)

target_link_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${ALL_LIB_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        ${ALL_LIBS}
        TBB::tbb                          # ← 链接 TBB 库
)
