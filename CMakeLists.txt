cmake_minimum_required(VERSION 3.31)
project(Game_of_Life LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")

#Asegura que los .cu se compilen como CUDA,
enable_language(CUDA)
include_directories("${CUDA_PATH}/include")
link_directories("${CUDA_PATH}/lib/x64")

add_executable(Game_of_Life
        src/main.cpp
        src/CpuLife/cpuLife.h
        src/CudaLife/cudaLife.h
        src/OpenclLife/openclLife.h
        src/CpuLife/cpuLife.cpp
        src/CudaLife/cudaLife.cpp
        src/CudaLife/cudaLife.cu
        src/OpenclLife/openclLife.cpp)


# Copia el kernel OpenCL al directorio de salida (build)
configure_file(src/OpenclLife/openclLife.cl openclLife.cl COPYONLY)

# Incluir headers de OpenCL
include_directories("C:/Users/Usuario/Desktop/U-Cursed/Semestre 11/Computación en GPU/OpenCL-SDK/include")

# Directorio donde está OpenCL.lib
link_directories("C:/Users/Usuario/Desktop/U-Cursed/Semestre 11/Computación en GPU/OpenCL-SDK/build/lib")

# Enlaza contra la librería OpenCL
target_link_libraries(Game_of_Life OpenCL)
