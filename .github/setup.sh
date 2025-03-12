#!/bin/bash
sudo apt install cmake
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
cd build_cpu
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` .
cmake --build . --config Release
./test-net
