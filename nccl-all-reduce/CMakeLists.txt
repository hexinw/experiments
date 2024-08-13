cmake_minimum_required(VERSION 3.10)
project(all_reduce_nccl_example)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(Torch_DIR /home/hexinw/workspace/thirdparty/pytorch/torch/share/cmake/Torch)
set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

include_directories(${Python3_INCLUDE_DIRS})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Include Gloo headers if needed
#include_directories(/home/hexinw/workspace/thirdparty/pytorch/third_party/gloo)

add_executable(all_reduce_nccl_example main.cpp)
target_link_libraries(all_reduce_nccl_example "${TORCH_LIBRARIES}")
set_property(TARGET all_reduce_nccl_example PROPERTY CXX_STANDARD 17)
