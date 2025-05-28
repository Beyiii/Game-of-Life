// src/OpenclLife/openclLife.cpp
#include "openclLife.h"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>

#define CHECK_CL(err, msg) if (err != CL_SUCCESS) { \
    std::cerr << "OpenCL error " << err << ": " << msg << std::endl; \
    std::exit(EXIT_FAILURE); \
}

std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void runGameOfLifeOpenCL(size_t width, size_t height, int iterations) {
    // 1) Mundo aleatorio en host
    std::vector<ubyte> h_world(width * height);
    std::srand((unsigned)std::time(nullptr));
    for (auto& cell : h_world)
        cell = std::rand() % 2;

    cl_int err;

    // 2) Plataforma y dispositivo
    cl_platform_id platform;
    cl_device_id device;
    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_CL(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_CL(err, "clGetDeviceIDs");

    // 3) Contexto y cola
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    // 4) Buffers
    size_t totalBytes = width * height * sizeof(ubyte);
    cl_mem d_data   = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalBytes, h_world.data(), &err);
    CHECK_CL(err, "clCreateBuffer d_data");
    cl_mem d_result = clCreateBuffer(context, CL_MEM_READ_WRITE, totalBytes, nullptr, &err);
    CHECK_CL(err, "clCreateBuffer d_result");

    // 5) Programa y kernel
    std::string source = loadKernelSource("openclLife.cl");
    const char* src = source.c_str();
    size_t length = source.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &err);
    CHECK_CL(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cl_kernel kernel = clCreateKernel(program, "game_of_life", &err);
    CHECK_CL(err, "clCreateKernel");

    // 6) Iteraciones
    size_t total = width * height;
    for (int i = 0; i < iterations; ++i) {
        err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_result);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &width);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &height);
        CHECK_CL(err, "clSetKernelArg");

        size_t globalSize = total;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        CHECK_CL(err, "clEnqueueNDRangeKernel");

        std::swap(d_data, d_result);
    }

    // 7) Resultado
    err = clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, totalBytes, h_world.data(), 0, nullptr, nullptr);
    CHECK_CL(err, "clEnqueueReadBuffer");

    // 8) Limpieza
    clReleaseMemObject(d_data);
    clReleaseMemObject(d_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // 9) Imprimir
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x)
            std::cout << (h_world[y * width + x] ? 'O' : '.');
        std::cout << '\n';
    }
}
