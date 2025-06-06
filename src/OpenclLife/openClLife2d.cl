// game_of_life_2d.cl
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef unsigned char uchar;

// Cuenta las 8 vecinas, igual que en CUDA, pero como función inlined
inline uchar countAliveCells2D(__global const uchar* data,
                               int x0, int x1, int x2,
                               int y0, int y1, int y2,
                               uint width) {
    return data[y0 * width + x0] + data[y0 * width + x1] + data[y0 * width + x2] +
           data[y1 * width + x0]                         + data[y1 * width + x2] +
           data[y2 * width + x0] + data[y2 * width + x1] + data[y2 * width + x2];
}

// Kernel 2D en OpenCL, cada work-item procesa (x,y)
__kernel void gameOfLifeKernel2D(__global const uchar* data,
                                 __global uchar* result,
                                 const uint width,
                                 const uint height) {
    // Obtenemos las coordenadas 2D
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= (int)width || y >= (int)height) return;

    // Coordenadas con wrap-around
    int x0 = (x + width  - 1) % width;
    int x2 = (x + 1) % width;
    int y0 = (y + height - 1) % height;
    int y2 = (y + 1) % height;

    // Contamos vecinas vivas
    uchar alive = countAliveCells2D(data, x0, x, x2, y0, y, y2, width);
    uchar center = data[y * width + x];
    result[y * width + x] = (alive == 3 || (alive == 2 && center)) ? 1 : 0;
}
