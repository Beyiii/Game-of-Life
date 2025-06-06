#include "cudaLife.h"
#include <cuda_runtime.h>
#include <cassert>
#include <algorithm>

// Cuenta las 8 vecinas (no incluye la celda central)
__device__ inline ubyte countAliveCells(const ubyte* data,
                                        int x0, int x1, int x2,
                                        int y0, int y1, int y2,
                                        size_t width) {
    return data[y0 * width + x0] + data[y0 * width + x1] + data[y0 * width + x2] +
           data[y1 * width + x0]                         + data[y1 * width + x2] +
           data[y2 * width + x0] + data[y2 * width + x1] + data[y2 * width + x2];
}

__global__ void gameOfLifeKernel(const ubyte* d_data,
                                 ubyte* d_result,
                                 size_t width, size_t height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int x  = idx % width;
    int y  = idx / width;
    int x0 = (x + width  - 1) % width;
    int x2 = (x + 1) % width;
    int y0 = (y + height - 1) % height;
    int y2 = (y + 1) % height;

    ubyte alive = countAliveCells(d_data, x0, x, x2, y0, y, y2, width);
    d_result[idx] = (alive == 3 || (alive == 2 && d_data[idx])) ? 1 : 0;
}

// Kernel 2D para el Juego de la Vida, cada thread procesa una celda (x,y)
__global__ void gameOfLifeKernel2D(const ubyte* d_data, ubyte* d_result, size_t width, size_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Coordenadas con borde periódico
    int x0 = (x + width  - 1) % width;
    int x2 = (x + 1) % width;
    int y0 = (y + height - 1) % height;
    int y2 = (y + 1) % height;

    // Cuenta las células vivas vecinas y aplica reglas del juego
    ubyte alive = countAliveCells(d_data, x0, x, x2, y0, y, y2, width);
    d_result[y * width + x] = (alive == 3 || (alive == 2 && d_data[y * width + x])) ? 1 : 0;
}



void computeIterationGPU(ubyte*& d_data,
                         ubyte*& d_result,
                         size_t width, size_t height, int threadsPerBlock) {
    size_t total = width * height;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    gameOfLifeKernel<<<blocks, threadsPerBlock>>>(d_data, d_result, width, height);
    cudaDeviceSynchronize();

    // swap para la siguiente iteración
    std::swap(d_data, d_result);
}

void computeIterationGPU2D(ubyte*& d_data, ubyte*& d_result, size_t width, size_t height) {
    dim3 threads(32, 32);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    gameOfLifeKernel2D<<<blocks, threads>>>(d_data, d_result, width, height);
    cudaDeviceSynchronize();
    std::swap(d_data, d_result);
}


void initWorld(ubyte*& d_data, ubyte*& d_result,
               size_t width, size_t height) {
    size_t bytes = width * height * sizeof(ubyte);
    cudaMalloc(&d_data,   bytes);
    cudaMalloc(&d_result, bytes);
}

void cleanupWorld(ubyte* d_data, ubyte* d_result) {
    cudaFree(d_data);
    cudaFree(d_result);
}