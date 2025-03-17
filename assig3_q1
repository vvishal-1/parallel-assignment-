/*
Q1. DAXPY Loop Using MPI
Formula: 𝑋[𝑖]=𝑎×𝑋[𝑖]+𝑌[𝑖]
X and 𝑌 are vectors of size 2^16
a is a scalar.
Compare MPI implementation speedup with a serial version.
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1 << 16)  // 2^16 elements

void daxpy_serial(double a, double *X, double *Y) {
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

void daxpy_parallel(double a, double *X, double *Y, int rank, int size) {
    int local_n = N / size;  // Divide elements among processes
    int start = rank * local_n;
    int end = start + local_n;

    for (int i = start; i < end; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double *X, *Y;
    double a = 2.5;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate vectors
    X = (double *)malloc(N * sizeof(double));
    Y = (double *)malloc(N * sizeof(double));

    // Initialize X and Y in root process
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            X[i] = rand() % 10;
            Y[i] = rand() % 10;
        }
    }

    // Broadcast X and Y to all processes
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    start_time = MPI_Wtime();
    daxpy_parallel(a, X, Y, rank, size);
    end_time = MPI_Wtime();

    // Gather results at rank 0
    MPI_Gather(X + rank * (N / size), N / size, MPI_DOUBLE, X, N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI DAXPY Time: %lf seconds\n", end_time - start_time);
    }

    free(X);
    free(Y);
    MPI_Finalize();
    return 0;
}
