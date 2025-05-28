#ifndef OPENCLLIFE_H
#define OPENCLLIFE_H

#include <cstddef>

typedef unsigned char ubyte;

// Ejecuta Game of Life usando OpenCL
void runGameOfLifeOpenCL(size_t width, size_t height, int iterations);

#endif // OPENCLLIFE_H
