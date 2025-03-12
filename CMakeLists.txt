cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(train-net LANGUAGES CXX CUDA)
cmake_policy(SET CMP0004 OLD)
find_package(Torch REQUIRED)

# Enable CUDA language support
find_package(CUDAToolkit REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${CUDAToolkit_CXX_FLAGS} -pthread")

add_executable(train-net source/train_net.cpp)
target_link_libraries(train-net  "${CUDAToolkit_libraries} ${TORCH_LIBRARIES}")
set_property(TARGET train-net PROPERTY CXX_STANDARD 17)

add_executable(test-net tests/test_net.cpp)
target_link_libraries(test-net  "${CUDAToolkit_libraries} ${TORCH_LIBRARIES}")
set_property(TARGET test-net PROPERTY CXX_STANDARD 17)