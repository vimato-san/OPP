#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>

#define SIZE 16500
#define EPSILON 0.0001
#define TAU 0.0001
#define RANK_ROOT 0
#define BEGIN_ACCURACY 1


void vector_print(double *vector, size_t vector_size) {
    printf("{");
    for (int i = 0; i < vector_size; i++)
        printf("%.3f, ", vector[i]);
    printf("\b\b}\n");
}


double vector_norm(const double *vector, size_t vector_size) {
    double sum = 0;
    for (int i = 0; i < vector_size; i++) {
        sum += (vector[i] * vector[i]);
    }
    return sqrt(sum);
}


void vector_diff(const double *vec_1, const double *vec_2, double *dest, size_t size) {
    for (int i = 0; i < size; i++)
        dest[i] = vec_1[i] - vec_2[i];
}


void vector_mult_scalar(const double *vector, double scalar, double *dest, size_t size) {
    for (int i = 0; i < size; i++)
        dest[i] = vector[i] * scalar;
}


void vector_mult_matrix(const double *vector, const double *matrix, double *result, size_t vector_size, int rank, int size) {
    int n_partial = (int)vector_size / size;
    int *send_counts;
    int *displs;

    if (rank == RANK_ROOT) {
        send_counts = (int *)malloc(sizeof(int) * size);
        displs = (int *)malloc(sizeof(int) * size);
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = n_partial * (int)vector_size;
            if (i < vector_size % size)
                send_counts[i] += (int)vector_size;
            if (i > 0)
                displs[i] = displs[i - 1] + send_counts[i - 1];
        }
    }

    if (rank < vector_size % size) {
        n_partial++;
    }
    double *a_partial = (double *)malloc(sizeof(double) * n_partial * vector_size);
    double *res_partial = (double *)malloc(sizeof(double) * n_partial);

    MPI_Scatterv(matrix, send_counts, displs, MPI_DOUBLE, a_partial,n_partial * (int)vector_size, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    double sum;
    for (int i = 0; i < n_partial; i++) {
        sum = 0;
        for (int j = 0; j < vector_size; j++)
            sum += a_partial[i * vector_size + j] * vector[j];
        res_partial[i] = sum;
    }

    int *recv_counts;
    if (rank == RANK_ROOT) {
        recv_counts = (int *)malloc(sizeof(int) * size);
        for (int i = 0; i < size; i++) {
            recv_counts[i] = send_counts[i] / (int)vector_size;
            if (i > 0)
                displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }
    MPI_Gatherv(res_partial, n_partial, MPI_DOUBLE, result, recv_counts, displs, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

    if (rank == RANK_ROOT) {
        free(send_counts);
        free(recv_counts);
        free(displs);
    }
    free(res_partial);
    free(a_partial);
}


void init_matrix(double *A, double *b, size_t N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            if (i == j)
                A[N * i + j] = 2;
            else
                A[N * i + j] = 1;
        b[i] = (int)N + 1;
    }
}

void init_vector(double *x, size_t N) {
    for (int i = 0; i < N; i++) {
        x[i] = 0;
    }
}


double vector_accuracy(double *Ax, double *b, size_t N) {
    double *tmp = (double *)malloc(sizeof(double) * N);
    vector_diff(Ax, b, tmp, N);
    double res = vector_norm(tmp, N) / vector_norm(b, N);

    free(tmp);
    return res;
}


int main(int argc, char *argv[]) {
    int size, rank;
    size_t N = SIZE;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    double *A;
    double *Ax;
    double *b;
    double *x = (double *)malloc(sizeof(double) * N);
    init_vector(x, N);
    if (rank == RANK_ROOT) {
        A = (double *)malloc(sizeof(double) * N * N);
        Ax = (double *)malloc(sizeof(double) * N);
        b = (double *)malloc(sizeof(double) * N);
        init_matrix(A, b, N);
    }
    double global_time = MPI_Wtime();
    vector_mult_matrix(x, A, Ax, N, rank, size);

    double accuracy = BEGIN_ACCURACY;
    while (accuracy > EPSILON) {
        double iteration_time = MPI_Wtime();
        if (rank == RANK_ROOT) {
            vector_diff(Ax, b, Ax, N);
            vector_mult_scalar(Ax, TAU, Ax, N);
            vector_diff(x, Ax, x, N);
        }
        MPI_Bcast(x, (int)N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
        vector_mult_matrix(x, A, Ax, N, rank, size);
        iteration_time = MPI_Wtime() - iteration_time;
        if (rank == RANK_ROOT) {
            accuracy = vector_accuracy(Ax, b, N);
            printf("time = %.5f, accuracy = %.10f\n", iteration_time, accuracy);
        }
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    }

    global_time = MPI_Wtime() - global_time;

    if (rank == RANK_ROOT) {
        free(Ax);
        free(A);
        free(b);

        vector_print(x, N);
        printf("time = %.5f\n", global_time);
    }
    free(x);
    MPI_Finalize();
    return 0;
}
