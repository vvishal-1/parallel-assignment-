// Parallel Prefix Sum (Scan) using MPI
// prefix_sum[i]=A[0]+A[1]+...+A[i]
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 8  // Number of elements

int main(int argc, char *argv[]) {
    int rank, size;
    int local_value, prefix_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize local values (each process gets one value)
    local_value = rank + 1; // Example: 1, 2, 3, ..., size

    // Perform parallel prefix sum using MPI_Scan
    MPI_Scan(&local_value, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("Process %d: Local Value = %d, Prefix Sum = %d\n", rank, local_value, prefix_sum);

    MPI_Finalize();
    return 0;
}
