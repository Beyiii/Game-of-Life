cmake_minimum_required(VERSION 3.31)
project(Game_of_Life)

set(CMAKE_CXX_STANDARD 14)

add_executable(Game_of_Life
        src/main.cpp
        src/CpuLife/cpuLife.h
        src/CudaLife/cudaLife.h
        src/OpenclLife/openclLife.h
        src/CpuLife/cpuLife.cpp)
