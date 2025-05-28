

// src/CudaLife/cudaLife.cpp
#include "cudaLife.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>

// Implementación de la función de conveniencia declarada en el .h
void runGameOfLifeGPU(size_t width, size_t height, int iterations) {
    // 1) Inicializa un mundo random en host
    std::vector<ubyte> h_world(width * height);
    std::srand((unsigned)std::time(nullptr));
    for (size_t i = 0; i < width * height; ++i)
        h_world[i] = std::rand() % 2;

    // 2) Reserva y copia a GPU
    ubyte *d_data = nullptr, *d_result = nullptr;
    initWorld(d_data, d_result, width, height);
    cudaMemcpy(d_data, h_world.data(),
               width * height * sizeof(ubyte),
               cudaMemcpyHostToDevice);

    // 3) Itera
    for (int it = 0; it < iterations; ++it) {
        computeIterationGPU(d_data, d_result, width, height);
    }

    // 4) Trae el resultado y limpia
    cudaMemcpy(h_world.data(), d_data,
               width * height * sizeof(ubyte),
               cudaMemcpyDeviceToHost);
    cleanupWorld(d_data, d_result);

    // 5) Imprime por consola
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x)
            std::cout << (h_world[y * width + x] ? 'O' : '.');
        std::cout << "\n";
    }
}
