#include "cpuLife.h"
#include <cstdlib>
#include <algorithm>
#include <ctime>

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
