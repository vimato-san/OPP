#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>


void printVector(double *A, size_t N) {
    for (int i = 0; i < N; i++)
        printf("%.3f, ", A[i]);
}


double normOfVector(const double *A, size_t N) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += (A[i] * A[i]);
    }
    return sqrt(sum);
}


void vectorDiff(double *A, const double *B, size_t N) {
    for (int i = 0; i < N; i++)
        A[i] -= B[i];
}


void vectorByNum(double *vector, double num, size_t N) {
    for (int i = 0; i < N; i++)
        vector[i] *= num;
}


double* vectorByMatrix(const double *A, const double *x, size_t N, int rank, int size) {
    double *res = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        if (rank == 0 && i % size != 0) {
            MPI_Recv(&res[i], 1, MPI_DOUBLE, MPI_ANY_SOURCE,
                         123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank == i % size) {
            double elem = 0;
            for (int j = 0; j < N; j++)
                elem += A[i * N + j] * x[j];
            if (rank != 0) {
                MPI_Send(&elem,1, MPI_DOUBLE, 0,
                         123, MPI_COMM_WORLD);
            }
            else
                res[i] = elem;
        }
    }
    MPI_Bcast(res, (int)N, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    return res;
}


void initValues(double *A, double *x, double *b, size_t N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            if (i == j)
                A[N * i + j] = 2;
            else
                A[N * i + j] = 1;
        x[i] = 0;
        b[i] = (double)N + 1;
    }
}


double vectorAccuracy(double *A, double *x, double *b, size_t N, int rank, int size) {
    double *Ax = vectorByMatrix(A, x, N, rank, size);
    vectorDiff(Ax, b, N);
    double res = normOfVector(Ax, N) / normOfVector(b, N);

    free(Ax);
    return res;
}


int main(int argc, char *argv[]) {
    int size,rank;
    size_t N = 5;
    double A[N*N];
    double b[N];
    double x[N];
    double epsilon = 0.01;
    double tau = 0.01;
    initValues(A, x, b, N);

    MPI_Init(&argc,&argv); // Инициализация MPI
    MPI_Comm_size(MPI_COMM_WORLD,&size); // Получение числа процессов
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); // Получение номера процесса

    double *Ax;
    while (vectorAccuracy(A, x, b, N, rank, size) > epsilon) {
        Ax = vectorByMatrix(A, x, N, rank, size);
        vectorDiff(Ax, b, N);
        vectorByNum(Ax, tau, N);
        vectorDiff(x, Ax, N);
    }
    free(Ax);

    if (rank == 0)
        printVector(x, N);
    MPI_Finalize();
    return 0;
}
