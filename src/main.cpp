#include <iostream>
#include <chrono>
#include "CpuLife/cpuLife.h"  // 👈 Ahora con la nueva estructura

int main() {
    const size_t width = 10000;
    const size_t height = 10000;
    const int iterations = 1;

    initLife(width, height);
    randomizeLife();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        computeIterationSerial();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    size_t totalCells = getEvaluatedCells();
    double seconds = duration.count();
    double cellsPerSecond = totalCells / seconds;

    std::cout << "Simulación de " << iterations << " iteraciones en "
              << width << "x" << height << " tomó "
              << seconds << " segundos.\n";
    std::cout << "Total de células evaluadas: " << totalCells << "\n";
    std::cout << "Células evaluadas por segundo: " << cellsPerSecond << "\n";

    freeLife();
    return 0;
}
