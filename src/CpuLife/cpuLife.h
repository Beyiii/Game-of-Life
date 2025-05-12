#ifndef CPULIFE_H
#define CPULIFE_H

#include <cstddef>

typedef unsigned char ubyte;

void initLife(size_t width, size_t height);
void randomizeLife();
void computeIterationSerial();
const ubyte* getLifeData();
void freeLife();
size_t getEvaluatedCells();

#endif // CPULIFE_H
