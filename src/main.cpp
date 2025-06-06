#include <iostream>
#include <fstream>
#include <vector>

#include "CpuLife/cpuLife.h"
#include "CudaLife/cudaLife.h"
#include "OpenclLife/openclLife.h"

// Devuelve la cantidad total de células evaluadas en una simulación
size_t getEvaluatedCells(size_t width, size_t height, int iterations) {
    return width * height * iterations;
}

void runOpenCL() {
    std::ofstream outFile("../resultadosOpenCL.txt");

    // Escribir cabecera
    outFile << "Width\tHeight\tIteraciones\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    // Lista de tamaños a probar
    std::vector<size_t> sizes = {500, 1000, 1500, 2000};
    int iterations = 10;

    // Para cada tamaño, correr el experimento y escribir resultados
    for (size_t size : sizes) {
        std::cout << "Running experiment for size " << size << "x" << size << "...\n";

        ExperimentResult result = runExperimentOpenCL(size, size, iterations, 128);

        outFile << result.width << '\t'
                << result.height << '\t'
                << result.iterations << '\t'
                << result.seconds << '\t'
                << result.totalCells << '\t'
                << result.cellsPerSecond << '\n';
    }

    outFile.close();

    std::cout << "Experiments completed. Results saved to resultadosOpenCL.txt\n";
}

void runOpenCLMultiplesOf32() {
    std::ofstream outFile("../resultadosOpenCL_multiples_32.txt");

    // Escribir cabecera
    outFile << "Width\tHeight\tIteraciones\tLocalWorkSize\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    // Lista de tamaños a probar
    std::vector<size_t> sizes = {500, 1000, 1500, 2000};
    int iterations = 10;

    // Lista de múltiplos de 32 para localWorkSize
    std::vector<size_t> localWorkSizes = {32, 64, 96, 128, 160};

    // Para cada tamaño
    for (size_t size : sizes) {
        // Para cada localWorkSize
        for (size_t lws : localWorkSizes) {
            std::cout << "Running MULTIPLE-OF-32 experiment for size " << size << "x" << size
                      << " with localWorkSize = " << lws << "...\n";
            ExperimentResult result = runExperimentOpenCL(size, size, iterations, lws);
            outFile << result.width << '\t'
                    << result.height << '\t'
                    << result.iterations << '\t'
                    << lws << '\t'
                    << result.seconds << '\t'
                    << result.totalCells << '\t'
                    << result.cellsPerSecond << '\n';
        }
    }
    outFile.close();
    std::cout << "Multiple-of-32 experiments completed. Results saved to resultadosOpenCL_multiples_32.txt\n";
}


void runOpenCLNonMultipleOf32() {
    std::ofstream outFile("../resultadosOpenCL_non_multiple_32.txt");

    // Escribir cabecera
    outFile << "Width\tHeight\tIteraciones\tLocalWorkSize\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    // Lista de tamaños a probar
    std::vector<size_t> sizes = {500, 1000, 1500, 2000};
    int iterations = 10;

    // Lista de localWorkSize no múltiplos de 32
    std::vector<size_t> localWorkSizes = {30, 150, 200, 150, 300};

    // Para cada tamaño
    for (size_t size : sizes) {
        // Para cada localWorkSize no múltiplo de 32
        for (size_t lws : localWorkSizes) {
            std::cout << "Running NON-MULTIPLE-OF-32 experiment for size " << size << "x" << size
                      << " with localWorkSize = " << lws << "...\n";

            ExperimentResult result = runExperimentOpenCL(size, size, iterations, lws);

            outFile << result.width << '\t'
                    << result.height << '\t'
                    << result.iterations << '\t'
                    << lws << '\t'
                    << result.seconds << '\t'
                    << result.totalCells << '\t'
                    << result.cellsPerSecond << '\n';
        }
    }

    outFile.close();

    std::cout << "Non-multiple-of-32 experiments completed. Results saved to resultadosOpenCL_non_multiple_32.txt\n";
}

void runOpenCL2D() {
    std::ofstream outFile("../resultadosOpenCL2D.txt");

    // Escribir cabecera
    outFile << "Width\tHeight\tIteraciones\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    // Lista de tamaños a probar
    std::vector<size_t> sizes = {500, 1000, 1500, 2000};
    int iterations = 10;
    size_t blockSize = 32; // ejemplo: bloque 32×32 (múltiplo de 32)

    // Para cada tamaño, correr el experimento 2D y escribir resultados
    for (size_t size : sizes) {
        std::cout << "Running 2D experiment for size " << size << "x" << size
                  << " with block " << blockSize << "x" << blockSize << "...\n";

        ExperimentResult result =
            runExperimentOpenCL2D(size, size, iterations, blockSize);

        outFile << result.width << '\t'
                << result.height << '\t'
                << result.iterations << '\t'
                << result.seconds << '\t'
                << result.totalCells << '\t'
                << result.cellsPerSecond << '\n';
    }

    outFile.close();
    std::cout << "2D experiments completed. Results saved to resultadosOpenCL2D.txt\n";
}


int main() {
    std::cout << "Ejecutando Game of Life con CPU...\n";
    runBenchmark();
    std::cout << "¡Experimentos de CPU termiandos sin errores!\n";

    std::cout << "Ejecutando Game of Life con CUDA...\n";
    runCuda();
    std::cout << "¡Experimentos de CUDA termiandos sin errores!\n";
    std::cout << "Ejecutando Game of Life con CUDA multiplos de 32...\n";
    runCudaMultiplesOf32();
    std::cout << "¡Experimentos de CUDA termiandos sin errores!\n";
    std::cout << "Ejecutando Game of Life con CUDA NO multiplos de 32...\n";
    runCudaNonMultiplesOf32();
    std::cout << "¡Experimentos de CUDA termiandos sin errores!\n";
    std::cout << "Ejecutando Game of Life con CUDA 2D...\n";
    runCuda2D();
    std::cout << "¡Experimentos de CUDA termiandos sin errores!\n";

    std::cout << "Ejecutando Game of Life con OpenCL...\n";
    runOpenCL();
    std::cout << "Ejecutando Game of Life con OpenCL (multiplos de 32)...\n";
    runOpenCLMultiplesOf32();
    std::cout << "Ejecutando Game of Life con OpenCL (no multiplos de 32)...\n";
    runOpenCLNonMultipleOf32();
    std::cout << "Ejecutando Game of Life con OpenCL2D ...\n";
    runOpenCL2D();
    std::cout << "¡Experimentos de OpenCL termiandos sin errores!\n";

    std::cout << "¡Todos los experimentos termiandos sin errores!\n";
    return 0;
}