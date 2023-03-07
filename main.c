#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>

#define SIZE 15000
#define EPSILON 0.0001
#define TAU 0.0001
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
    double *res;
    res = (double *)malloc(sizeof(double) * N);
    // n_partial для каждого процесса свой
    int n_partial = (int)N / size;
    int *send_counts;
    int *displs;

    if (rank == RANK_ROOT) {
        send_counts = (int *)malloc(sizeof(int) * size);
        displs = (int *)malloc(sizeof(int) * size);
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = n_partial * (int)N;
            if (i < N % size)
                send_counts[i] += (int)N;
            if (i > 0)
                displs[i] = displs[i - 1] + send_counts[i - 1];
        }
    }

    if (rank < N % size) {
        n_partial++;
    }
    double *a_partial = (double *)malloc(sizeof(double) * n_partial * N);
    double *res_partial = (double *)malloc(sizeof(double) * n_partial);

    MPI_Scatterv(A, send_counts, displs, MPI_DOUBLE, a_partial,n_partial * (int)N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    double sum;
    for (int i = 0; i < n_partial; i++) {
        sum = 0;
        for (int j = 0; j < N; j++)
            sum += a_partial[i * N + j] * x[j];
        res_partial[i] = sum;
    }

    int *recv_counts;
    if (rank == RANK_ROOT) {
        recv_counts = (int *)malloc(sizeof(int) * size);
        for (int i = 0; i < size; i++) {
            recv_counts[i] = send_counts[i] / (int)N;
            if (i > 0)
                displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }
    MPI_Gatherv(res_partial, n_partial, MPI_DOUBLE, res, recv_counts, displs, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    MPI_Bcast(res, (int)N, MPI_DOUBLE, RANK_ROOT,MPI_COMM_WORLD);
    if (rank == RANK_ROOT) {
        free(send_counts);
        free(recv_counts);
        free(displs);
    }
    free(res_partial);
    free(a_partial);
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
    size_t N = SIZE;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    double *A;
    double *b = (double *)malloc(sizeof(double) * N);
    double *x = (double *)malloc(sizeof(double) * N);
    init_vectors(x, b, N);
    if (rank == RANK_ROOT) {
        A = (double *)malloc(sizeof(double) * N * N);
        init_matrix(A, N);
    }
    double t = MPI_Wtime();
    double *Ax = vector_mult_matrix(A, x, N, rank, size);

    double accuracy = vector_accuracy(Ax, b, N);
    while (accuracy > EPSILON) {
        double t2 = MPI_Wtime();

        vector_diff(Ax, b, N);
        vector_mult_scalar(Ax, TAU, N);
        vector_diff(x, Ax, N);
        free(Ax);
        Ax = vector_mult_matrix(A, x, N, rank, size);
        accuracy = vector_accuracy(Ax, b, N);
        t2 = MPI_Wtime() - t2;
        if (rank == RANK_ROOT) {
            printf("time = %.5f, accuracy = %.10f\n", t2, accuracy);
        }
    }
    free(Ax);

    t = MPI_Wtime() - t;

    if (rank == RANK_ROOT) {
        free(A);
        vector_print(x, N);
        printf("time = %.5f\n", t);
    }
    free(b);
    free(x);
    MPI_Finalize();
    return 0;
}
