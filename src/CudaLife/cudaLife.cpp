

// src/CudaLife/cudaLife.cpp
#include "cudaLife.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>

// Implementación de la función de conveniencia declarada en el .h
void runGameOfLifeCuda(size_t width, size_t height, int iterations) {
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
        computeIterationGPU(d_data, d_result, width, height, 128);
    }

    // 4) Trae el resultado y limpia
    cudaMemcpy(h_world.data(), d_data,
               width * height * sizeof(ubyte),
               cudaMemcpyDeviceToHost);
    cleanupWorld(d_data, d_result);
}

// Llena un arreglo con valores aleatorios 0 o 1 usando distribución de Bernoulli
void fillRandom(ubyte* data, size_t width, size_t height) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(0.5);
    for (size_t i = 0; i < width * height; ++i)
        data[i] = d(gen);
}

// Experimento 1: mide el tiempo de ejecución en GPU con configuración por defecto
void runCuda() {
    const std::string outputFile = "../resultados_cuda.txt";
    std::ofstream output(outputFile);
    if (!output.is_open()) {
        std::cerr << "Error al abrir " << outputFile << "\n";
        return;
    }

    output << "Width\tHeight\tIteraciones\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    const size_t startSize = 500;
    const size_t maxSize = 2000;
    const size_t stepSize = 500;
    const int iterations = 10;

    for (size_t size = startSize; size <= maxSize; size += stepSize) {
        size_t width = size, height = size;
        size_t totalCells = width * height * iterations;

        auto start = std::chrono::high_resolution_clock::now();
        runGameOfLifeCuda(width, height, iterations);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double cellsPerSecond = totalCells / seconds;

        std::cout << "CUDA " << width << "x" << height
                  << " (" << iterations << " iters): "
                  << seconds << "s, " << cellsPerSecond << " celdas/s\n";

        output << width << "\t" << height << "\t"
               << iterations << "\t" << seconds << "\t"
               << totalCells << "\t" << cellsPerSecond << "\n";
    }

    output.close();
    std::cout << "\nResultados CUDA guardados en " << outputFile << "\n";
}

void runCudaMultiplesOf32() {
    const std::string outputFile = "../resultados_cuda_multiblock.txt";
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Error al abrir " << outputFile << "\n";
        return;
    }

    out << "Width\tHeight\tIteraciones\tTPB\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    std::vector<int> blockSizes = {32, 64, 96, 128, 160};  // Múltiplos de 32
    const size_t startSize = 500;
    const size_t maxSize = 2000;
    const size_t stepSize = 500;
    const int iterations = 10;

    for (size_t size = startSize; size <= maxSize; size += stepSize) {
        size_t width = size;
        size_t height = size;
        size_t totalCells = width * height * iterations;

        for (int TPB : blockSizes) {
            ubyte *h_data = new ubyte[width * height];
            fillRandom(h_data, width, height);

            ubyte *d_data, *d_result;
            initWorld(d_data, d_result, width, height);
            cudaMemcpy(d_data, h_data, width * height * sizeof(ubyte), cudaMemcpyHostToDevice);

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                computeIterationGPU(d_data, d_result, width, height, TPB);
            }
            auto end = std::chrono::high_resolution_clock::now();

            double seconds = std::chrono::duration<double>(end - start).count();
            double cellsPerSecond = totalCells / seconds;

            std::cout << "CUDA múltiplos de 32 " << width << "x" << height
                      << " | TPB=" << TPB
                      << " | " << seconds << "s, "
                      << cellsPerSecond << " celdas/s\n";

            out << width << "\t" << height << "\t" << iterations << "\t"
                << TPB << "\t" << seconds << "\t"
                << totalCells << "\t" << cellsPerSecond << "\n";

            delete[] h_data;
            cleanupWorld(d_data, d_result);
        }
    }

    out.close();
    std::cout << "\nResultados CUDA con bloques múltiplos de 32 guardados en "
              << outputFile << "\n";
}


void runCudaNonMultiplesOf32() {
    const std::string outputFile = "../resultados_cuda_notmultiblock.txt";
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Error al abrir " << outputFile << "\n";
        return;
    }

    out << "Width\tHeight\tIteraciones\tTPB\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    std::vector<int> blockSizes = {100, 150, 200, 250, 300};  // No múltiplos de 32
    const size_t startSize = 500;
    const size_t maxSize = 2000;
    const size_t stepSize = 500;
    const int iterations = 10;

    for (size_t size = startSize; size <= maxSize; size += stepSize) {
        size_t width = size;
        size_t height = size;
        size_t totalCells = width * height * iterations;

        for (int TPB : blockSizes) {
            ubyte *h_data = new ubyte[width * height];
            fillRandom(h_data, width, height);

            ubyte *d_data, *d_result;
            initWorld(d_data, d_result, width, height);
            cudaMemcpy(d_data, h_data, width * height * sizeof(ubyte), cudaMemcpyHostToDevice);

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                computeIterationGPU(d_data, d_result, width, height, TPB);
            }
            auto end = std::chrono::high_resolution_clock::now();

            double seconds = std::chrono::duration<double>(end - start).count();
            double cellsPerSecond = totalCells / seconds;

            std::cout << "CUDA (no mult de 32) " << width << "x" << height
                      << " | TPB=" << TPB
                      << " | " << seconds << "s, "
                      << cellsPerSecond << " celdas/s\n";

            out << width << "\t" << height << "\t" << iterations << "\t"
                << TPB << "\t" << seconds << "\t"
                << totalCells << "\t" << cellsPerSecond << "\n";

            delete[] h_data;
            cleanupWorld(d_data, d_result);
        }
    }

    out.close();
    std::cout << "\nResultados CUDA (no múltiplos de 32) guardados en "
              << outputFile << "\n";
}


void runCuda2D() {
    const std::string outputFile = "../resultados_cuda_2d.txt";
    std::ofstream output(outputFile);
    if (!output.is_open()) {
        std::cerr << "Error al abrir " << outputFile << "\n";
        return;
    }

    output << "Width\tHeight\tIteraciones\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    const size_t startSize = 500;
    const size_t maxSize = 2000;
    const size_t stepSize = 500;
    const int iterations = 10;

    for (size_t size = startSize; size <= maxSize; size += stepSize) {
        size_t width = size;
        size_t height = size;
        size_t totalCells = width * height * iterations;

        // Crear datos aleatorios
        ubyte *h_data = new ubyte[width * height];
        fillRandom(h_data, width, height);

        // Reservar memoria en GPU
        ubyte *d_data, *d_result;
        initWorld(d_data, d_result, width, height);
        cudaMemcpy(d_data, h_data, width * height * sizeof(ubyte), cudaMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            computeIterationGPU2D(d_data, d_result, width, height);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double cellsPerSecond = totalCells / seconds;

        std::cout << "CUDA 2D " << width << "x" << height
                  << " (" << iterations << " iters): "
                  << seconds << "s, " << cellsPerSecond << " celdas/s\n";

        output << width << "\t" << height << "\t"
               << iterations << "\t" << seconds << "\t"
               << totalCells << "\t" << cellsPerSecond << "\n";

        delete[] h_data;
        cleanupWorld(d_data, d_result);
    }

    output.close();
    std::cout << "\nResultados CUDA 2D guardados en " << outputFile << "\n";
}
