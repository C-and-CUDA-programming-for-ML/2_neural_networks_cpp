cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(train-net LANGUAGES CXX)
cmake_policy(SET CMP0004 OLD)
find_package(Torch REQUIRED)

add_executable(train-net ../source/train_net.cpp)
target_link_libraries(train-net  "${TORCH_LIBRARIES}")
set_property(TARGET train-net PROPERTY CXX_STANDARD 17)

add_executable(test-net ../tests/test_net.cpp)
target_link_libraries(test-net  "${TORCH_LIBRARIES}")
set_property(TARGET test-net PROPERTY CXX_STANDARD 17)
