

#ifndef CUDALIFE_H
#define CUDALIFE_H

#include <cstddef>
typedef unsigned char ubyte;

// Reserva y libera buffers en GPU
void initWorld(ubyte*& d_data, ubyte*& d_result,
               size_t width, size_t height);
void cleanupWorld(ubyte* d_data, ubyte* d_result);

// Lanza una iteración del kernel y hace swap de punteros
void computeIterationGPU(ubyte*& d_data,
                         ubyte*& d_result,
                         size_t width, size_t height);

// Función de conveniencia para inicializar, iterar y limpiar
void runGameOfLifeGPU(size_t width, size_t height, int iterations);

#endif // CUDALIFE_H