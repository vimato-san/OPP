#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>

#define EPSILON 0.00001
#define TAU 0.01
#define RANK_ROOT 0


void printVector(double *A, size_t N) {
    printf("{");
    for (int i = 0; i < N; i++)
        printf("%.3f, ", A[i]);
    printf("}\n");
}


double *vectorCopy(const double *A, size_t N) {
    double *copy = (double *)malloc(sizeof(double) * N);
    for (size_t i = 0; i < N; i++) {
        copy[i] = A[i];
    }
    return copy;
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


void vectorByScalar(double *vector, double num, size_t N) {
    for (int i = 0; i < N; i++)
        vector[i] *= num;
}


double* vectorByMatrix(const double *A, const double *x, size_t N, int rank, int size) {
    double *res = (double*)malloc(sizeof(double) * N);
    int n_partial = (int)N / size;
    double *a_partial = (double *) malloc(n_partial * N * sizeof(double));
    //Each process get equal number of rows to calculate
    MPI_Scatter(A, n_partial * (int)N, MPI_DOUBLE, a_partial, n_partial * (int)N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

    double sum;
    double *res_partial = (double *) malloc(n_partial * sizeof(double));
    for (int i = 0; i < n_partial; i++) {
        sum = 0;
        for (int j = 0; j < N; j++)
            sum += a_partial[i * N + j] * x[j];
        res_partial[i] = sum;
    }
    MPI_Gather(res_partial, n_partial, MPI_DOUBLE, res, n_partial, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

    //Calculate remaining rows of matrix
    for (int i = n_partial * size; i < N; i++) {
        if (rank == 0 && i % size != 0) {
            MPI_Recv(&res[i], 1, MPI_DOUBLE, MPI_ANY_SOURCE,
                         123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank == i % size) {
            sum = 0;
            for (int j = 0; j < N; j++)
                sum += A[i * N + j] * x[j];
            if (rank != 0) {
                MPI_Send(&sum,1, MPI_DOUBLE, RANK_ROOT,
                         123, MPI_COMM_WORLD);
            }
            else
                res[i] = sum;
        }
    }
    MPI_Bcast(res, (int)N, MPI_DOUBLE, RANK_ROOT,MPI_COMM_WORLD);
    free(res_partial);
    free(a_partial);
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


double vectorAccuracy(double *Ax, double *b, size_t N) {
    double *AxCopy = vectorCopy(Ax, N);
    vectorDiff(AxCopy, b, N);
    double res = normOfVector(AxCopy, N) / normOfVector(b, N);

    free(AxCopy);
    return res;
}


int main(int argc, char *argv[]) {
    int size,rank;
    size_t N = 3;
    double A[N*N];
    double b[N];
    double x[N];
    initValues(A, x, b, N);

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    double *Ax = vectorByMatrix(A, x, N, rank, size);
    while (vectorAccuracy(Ax, b, N) > EPSILON) {
        vectorDiff(Ax, b, N);
        vectorByScalar(Ax, TAU, N);
        vectorDiff(x, Ax, N);
        Ax = vectorByMatrix(A, x, N, rank, size);
    }
    free(Ax);

    if (rank == 0)
        printVector(x, N);
    MPI_Finalize();
    return 0;
}
