

#ifndef CUDALIFE_H
#define CUDALIFE_H

#include <cstddef>
#include <string>
typedef unsigned char ubyte;

// Reserva y libera buffers en GPU
void initWorld(ubyte*& d_data, ubyte*& d_result,
               size_t width, size_t height);
void cleanupWorld(ubyte* d_data, ubyte* d_result);

// Lanza una iteración del kernel y hace swap de punteros
void computeIterationGPU(ubyte*& d_data,
                         ubyte*& d_result,
                         size_t width, size_t height,
                         int threadsPerBlock);
void computeIterationGPU2D(ubyte*& d_data,
                           ubyte*& d_result,
                           size_t width, size_t height);

// Función de conveniencia para inicializar, iterar y limpiar
void runGameOfLifeCuda(size_t width, size_t height, int iterations);
void runCuda();
void runCudaMultiplesOf32();
void runCudaNonMultiplesOf32();
void runCuda2D();

#endif // CUDALIFE_H