#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;

    cudaError_t res = cudaGetDeviceProperties(&p, 0);
    if (res != cudaSuccess) {
        std::cerr << cudaGetErrorString(res);
        return 1;
    }

    std::cout << "name: " << p.name << std::endl;
    std::cout << "cc: " << p.major << "." << p.minor << std::endl;
    std::cout << "total global memory: " << p.totalGlobalMem / (1 << 20) << "MiB" << std::endl;
}
