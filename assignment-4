#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// CUDA kernel where each thread performs a different task
__global__ void compute_sums(int *iterative_sum, int *formula_sum) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Iterative approach
        int sum = 0;
        for (int i = 1; i <= N; ++i) {
            sum += i;
        }
        *iterative_sum = sum;
    } else if (tid == 1) {
        // Direct formula approach
        *formula_sum = N * (N + 1) / 2;
    }
}

int main() {
    int h_iterative_sum = 0;
    int h_formula_sum = 0;
    int *d_iterative_sum, *d_formula_sum;

    // Allocate memory on the device
    cudaMalloc((void**)&d_iterative_sum, sizeof(int));
    cudaMalloc((void**)&d_formula_sum, sizeof(int));

    // Launch kernel with 1 block and 2 threads
    compute_sums<<<1, 2>>>(d_iterative_sum, d_formula_sum);

    // Copy results back to host
    cudaMemcpy(&h_iterative_sum, d_iterative_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_formula_sum, d_formula_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // Display the results
    printf("Sum using iterative approach: %d\n", h_iterative_sum);
    printf("Sum using formula approach: %d\n", h_formula_sum);

    // Free device memory
    cudaFree(d_iterative_sum);
    cudaFree(d_formula_sum);

    return 0;
}
