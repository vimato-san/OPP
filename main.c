#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>

#define EPSILON 0.01
#define TAU 0.001
#define RANK_ROOT 0


void vector_print(double *A, size_t N) {
    printf("{");
    for (int i = 0; i < N; i++)
        printf("%.3f, ", A[i]);
    printf("\b\b}\n");
}


double *vector_copy(const double *A, size_t N) {
    double *copy = (double *)malloc(sizeof(double) * N);
    for (size_t i = 0; i < N; i++) {
        copy[i] = A[i];
    }
    return copy;
}


double vector_norm(const double *A, size_t N) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += (A[i] * A[i]);
    }
    return sqrt(sum);
}


void vector_diff(double *A, const double *B, size_t N) {
    for (int i = 0; i < N; i++)
        A[i] -= B[i];
}


void vector_mult_scalar(double *vector, double num, size_t N) {
    for (int i = 0; i < N; i++)
        vector[i] *= num;
}


double* vector_mult_matrix(const double *A, const double *x, size_t N, int rank, int size) {
    double *res = (double*)malloc(sizeof(double) * N);
    int n_partial = (int)N / size;
    double *a_partial = (double*)malloc(sizeof(double) * n_partial * N);
    double *x_partial = (double*)malloc(sizeof(double) * n_partial);
    double *res_partial = (double *) malloc(sizeof(double) * n_partial);


    //Each process get equal number of rows to calculate
    MPI_Scatter(A, n_partial * (int)N, MPI_DOUBLE, a_partial, n_partial * (int)N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    MPI_Scatter(x, n_partial, MPI_DOUBLE, x_partial, n_partial, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
//    vector_print(a_partial, n_partial * (int)N);
//    vector_print(x_partial, n_partial);

    double sum;
    for (int i = 0; i < n_partial; i++) {
        sum = 0;
        for (int j = 0; j < N; j++)
            sum += a_partial[i * N + j] * x_partial[i];
        res_partial[i] = sum;
    }
//    vector_print(res_partial, n_partial);

    MPI_Gather(res_partial, n_partial, MPI_DOUBLE, res, n_partial, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    /*if (rank == RANK_ROOT)
        vector_print(res, N);*/
    //Calculate remaining rows of matrix
    /*for (int i = n_partial * size; i < N; i++) {
        if (rank == RANK_ROOT && i % size != 0) {
            MPI_Recv(&res[i], 1, MPI_DOUBLE, MPI_ANY_SOURCE,
                         123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank == i % size) {
            sum = 0;
            for (int j = 0; j < N; j++)
                sum += A[i * N + j] * x[j];
            if (rank != RANK_ROOT) {
                MPI_Send(&sum,1, MPI_DOUBLE, RANK_ROOT,
                         123, MPI_COMM_WORLD);
            }
            else
                res[i] = sum;
        }
    }*/
    MPI_Bcast(res, (int)N, MPI_DOUBLE, RANK_ROOT,MPI_COMM_WORLD);
    free(res_partial);
    free(a_partial);
    free(x_partial);
    return res;
}


void init_matrix(double *A, size_t N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            if (i == j)
                A[N * i + j] = 2;
            else
                A[N * i + j] = 1;
    }
}

void init_vectors(double *x, double *b, size_t N) {
    for (int i = 0; i < N; i++) {
        x[i] = 0;
        b[i] = (double)N + 1;
    }
}


double vector_accuracy(double *Ax, double *b, size_t N) {
    double *Ax_copy = vector_copy(Ax, N);
    vector_diff(Ax_copy, b, N);
    double res = vector_norm(Ax_copy, N) / vector_norm(b, N);

    free(Ax_copy);
    return res;
}


int main(int argc, char *argv[]) {
    int size, rank;
    size_t N = 1900;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    double *A;
    double *b;
    double *x;
    b = (double *)malloc(sizeof(double) * N);
    x = (double *)malloc(sizeof(double) * N);
    init_vectors(x, b, N);
    if (rank == RANK_ROOT) {
        A = (double *)malloc(sizeof(double) * N * N);
        init_matrix(A, N);
        /*vector_print(b, N);
        vector_print(x, N);
        vector_print(A, N*N);*/
    }
    double t = MPI_Wtime();
    //printf("Hello, from proc: %d\n", rank);

    double *Ax = vector_mult_matrix(A, x, N, rank, size);

    while (vector_accuracy(Ax, b, N) > EPSILON) {
        vector_diff(Ax, b, N);
        vector_mult_scalar(Ax, TAU, N);
        vector_diff(x, Ax, N);
        free(Ax);
        Ax = vector_mult_matrix(A, x, N, rank, size);
    }
    free(Ax);

    t = MPI_Wtime() - t;

    if (rank == RANK_ROOT) {
        vector_print(x, N);
        printf("time = %.5f\n", t);
    }
    MPI_Finalize();
    return 0;
}
