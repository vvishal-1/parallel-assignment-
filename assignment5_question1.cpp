#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Statically defined global device arrays
__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

// Kernel function for vector addition
__global__ void vectorAdd() {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

int main() {
    float h_A[N], h_B[N], h_C[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Copy data from host to device
    cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    cudaEventRecord(start, 0);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>();

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // in milliseconds

    // Copy result from device to host
    cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

    // Verify result
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Error at index %d: %f + %f != %f\n", i, h_A[i], h_B[i], h_C[i]);
            return -1;
        }
    }

    printf("Vector addition successful.\n");
    printf("Kernel execution time: %f ms\n", elapsedTime);

    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Memory clock rate in KHz, bus width in bits
    int memClockRate = prop.memoryClockRate;
    int memBusWidth = prop.memoryBusWidth;

    // Calculate theoretical bandwidth in GB/s
    double bandwidth = 2.0 * memClockRate * (memBusWidth / 8.0) / 1e6;
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", bandwidth);

    // Calculate measured bandwidth
    float totalBytes = N * (2 + 1) * sizeof(float); // 3 * N * 4 bytes
    double elapsedTimeSec = elapsedTime / 1000.0; // Convert ms to seconds
    double measuredBW = totalBytes / (elapsedTimeSec * 1e9); // GB/s
    printf("Measured Memory Bandwidth: %.2f GB/s\n", measuredBW);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
