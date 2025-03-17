// 6. Parallel Dot Product using MPI
// dot_product=A0*B0+A1*B1+...+An*Bn​

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 100  // Size of vectors

int main(int argc, char *argv[]) {
    int rank, size;
    double local_dot = 0.0, global_dot = 0.0;
    double A[VECTOR_SIZE], B[VECTOR_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = VECTOR_SIZE / size;  // Divide work among processes

    // Root process initializes vectors
    if (rank == 0) {
        srand(0);
        for (int i = 0; i < VECTOR_SIZE; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
    }

    // Scatter vectors to all processes
    double local_A[elements_per_proc], local_B[elements_per_proc];
    MPI_Scatter(A, elements_per_proc, MPI_DOUBLE, local_A, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, elements_per_proc, MPI_DOUBLE, local_B, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process computes its local dot product
    for (int i = 0; i < elements_per_proc; i++) {
        local_dot += local_A[i] * local_B[i];
    }

    // Reduce all local dot products to get the final result at rank 0
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the final dot product
    if (rank == 0) {
        printf("Dot Product: %lf\n", global_dot);
    }

    MPI_Finalize();
    return 0;
}
