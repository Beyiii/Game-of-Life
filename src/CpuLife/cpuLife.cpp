#include "cpuLife.h"
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <chrono>

static ubyte* m_data = nullptr;
static ubyte* m_resultData = nullptr;
static size_t m_worldWidth = 0;
static size_t m_worldHeight = 0;
static size_t m_dataLength = 0;

static size_t totalEvaluatedCells = 0;

void initLife(size_t width, size_t height) {
    m_worldWidth = width;
    m_worldHeight = height;
    m_dataLength = width * height;

    m_data = new ubyte[m_dataLength];
    m_resultData = new ubyte[m_dataLength];
    totalEvaluatedCells = 0;
}

void randomizeLife() {
    srand((unsigned) time(nullptr));
    for (size_t i = 0; i < m_dataLength; ++i) {
        m_data[i] = rand() % 2;
    }
}

inline ubyte countAliveCells(size_t x0, size_t x1, size_t x2,
                             size_t y0, size_t y1, size_t y2) {
    return m_data[x0 + y0] + m_data[x1 + y0] + m_data[x2 + y0]
         + m_data[x0 + y1] + m_data[x2 + y1]
         + m_data[x0 + y2] + m_data[x1 + y2] + m_data[x2 + y2];
}

void computeIterationSerial() {
    for (size_t y = 0; y < m_worldHeight; ++y) {
        size_t y0 = ((y + m_worldHeight - 1) % m_worldHeight) * m_worldWidth;
        size_t y1 = y * m_worldWidth;
        size_t y2 = ((y + 1) % m_worldHeight) * m_worldWidth;

        for (size_t x = 0; x < m_worldWidth; ++x) {
            size_t x0 = (x + m_worldWidth - 1) % m_worldWidth;
            size_t x2 = (x + 1) % m_worldWidth;

            ubyte aliveCells = countAliveCells(x0, x, x2, y0, y1, y2);
            m_resultData[y1 + x] =
                aliveCells == 3 || (aliveCells == 2 && m_data[x + y1]) ? 1 : 0;
            ++totalEvaluatedCells;
        }
    }
    std::swap(m_data, m_resultData);
}

const ubyte* getLifeData() {
    return m_data;
}

size_t getEvaluatedCells() {
    return totalEvaluatedCells;
}

void freeLife() {
    delete[] m_data;
    delete[] m_resultData;
    m_data = nullptr;
    m_resultData = nullptr;
}

void runBenchmark() {
    // Abrimos archivo de salida
    std::ofstream output("../resultados.txt");
    if (!output.is_open()) {
        std::cerr << "Error al abrir resultados.txt\n";
        return;
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
}
