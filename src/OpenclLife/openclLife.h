#ifndef OPENCLLIFE_H
#define OPENCLLIFE_H

#include <cstddef>

typedef unsigned char ubyte;

// Estructura para devolver resultados de un experimento
struct ExperimentResult {
    size_t width;
    size_t height;
    int iterations;
    double seconds;
    size_t totalCells;
    double cellsPerSecond;
};

// Ejecuta Game of Life usando OpenCL
void runGameOfLifeOpenCL(size_t width, size_t height, int iterations, size_t localWorkSize);

// Ejecuta el experimento y retorna los resultados
ExperimentResult runExperimentOpenCL(size_t width, size_t height, int iterations, size_t localWorkSize);

void runGameOfLifeOpenCL2D(size_t width, size_t height, int iterations, size_t localWorkSize);

ExperimentResult runExperimentOpenCL2D(size_t width, size_t height, int iterations, size_t localWorkSize);

#endif // OPENCLLIFE_H
