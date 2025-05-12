#include <iostream>
#include <chrono>
#include <fstream>
#include "CpuLife/cpuLife.h"

// Devuelve la cantidad total de células evaluadas en una simulación
size_t getEvaluatedCells(size_t width, size_t height, int iterations) {
    return width * height * iterations;
}

int main() {
    // Abrimos archivo de salida
    std::ofstream output("../resultados.txt");
    if (!output.is_open()) {
        std::cerr << "Error al abrir resultados.txt\n";
        return 1;
    }

    // Cabecera del archivo
    output << "Width\tHeight\tIteraciones\tSegundos\tTotalCeldas\tCeldasPorSegundo\n";

    // Parámetros del experimento
    const size_t startSize = 500;
    const size_t maxSize = 2000;
    const size_t stepSize = 500;
    const int iterations = 10;

    for (size_t size = startSize; size <= maxSize; size += stepSize) {
        size_t width = size;
        size_t height = size;

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

        // Mostrar por consola
        std::cout << "Simulación de " << iterations << " iteraciones en "
                  << width << "x" << height << " tomó "
                  << seconds << " segundos.\n";

        // Guardar en archivo
        output << width << "\t"
               << height << "\t"
               << iterations << "\t"
               << seconds << "\t"
               << totalCells << "\t"
               << cellsPerSecond << "\n";

        freeLife();
    }

    output.close();
    std::cout << "\nResultados guardados en resultados.txt\n";

    return 0;
}