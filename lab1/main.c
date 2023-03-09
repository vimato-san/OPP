#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>

#define SIZE 11000
#define EPSILON 0.00001
#define TAU 0.00001
#define RANK_ROOT 0
#define BEGIN_ACCURACY 1


// args in cmd line: mpirun –np 1 cmake-build-debug/lab1

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


void vector_sub(const double *vec_1, const double *vec_2, double *dest, size_t size) {
    for (int i = 0; i < size; i++)
        dest[i] = vec_1[i] - vec_2[i];
}


void vector_mult_scalar(const double *vector, double scalar, double *dest, size_t size) {
    for (int i = 0; i < size; i++)
        dest[i] = vector[i] * scalar;
}


void vector_mult_matrix(const double *vector, size_t vector_size, const double *chunk_matrix, size_t chunk_size, double *result, int rank, int size) {
    double *res_partial = (double *)malloc(sizeof(double) * chunk_size);
    for (int i = 0; i < chunk_size; i++) {
        res_partial[i] = 0;
        for (int j = 0; j < vector_size; j++)
            res_partial[i] += chunk_matrix[i * vector_size + j] * vector[j];
    }

    // Filling arrays for MPI_Gatherv function
    int *recv_counts;
    int *displs;
    if (rank == RANK_ROOT) {
        recv_counts = (int *)malloc(sizeof(int) * size);
        displs = (int *)malloc(sizeof(int) * size);
        displs[0] = 0;
        int buff_size = (int)vector_size / size;
        for (int i = 0; i < size; i++) {
            recv_counts[i] = buff_size;
            if (i < vector_size % size)
                recv_counts[i]++;
            if (i > 0)
                displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }
    // Collect parts of result vector in root process
    MPI_Gatherv(res_partial, (int)chunk_size, MPI_DOUBLE, result, recv_counts, displs, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    if (rank == RANK_ROOT) {
        free(recv_counts);
        free(displs);
    }
    free(res_partial);
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


void vector_copy(const double *original, double *copy, size_t vector_size) {
    for (int i = 0; i < vector_size; i++)
        copy[i] = original[i];
}


double vector_accuracy(double *Ax, double b_norm, size_t N) {
    double *tmp = (double *)malloc(sizeof(double) * N);
    vector_copy(Ax, tmp, N);
    double res = vector_norm(tmp, N) / b_norm;

    free(tmp);
    return res;
}



void distribute_matrix(double *matrix, int vector_size, double *chunk_buff, int chunk_size, int rank, int size) {
    int *send_counts;
    int *displs;

    // Filling arrays for MPI_Scatterv function
    if (rank == RANK_ROOT) {
        send_counts = (int *)malloc(sizeof(int) * size);
        displs = (int *)malloc(sizeof(int) * size);
        int buff_size = vector_size / size;
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = buff_size * vector_size;
            if (i < vector_size % size)
                send_counts[i] += vector_size;
            if (i > 0)
                displs[i] = displs[i - 1] + send_counts[i - 1];
        }
    }
    MPI_Scatterv(matrix, send_counts, displs, MPI_DOUBLE, chunk_buff, chunk_size * vector_size, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    if (rank == RANK_ROOT) {
        free(send_counts);
        free(displs);
    }
}


int main(int argc, char *argv[]) {
    int size, rank;
    int N = SIZE;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    double *A;
    double *Ax;
    double *b;
    double *x = (double *)malloc(sizeof(double) * N);
    init_vector(x, N);
    double b_norm;  // Const norm of vector b

    double program_start_time = MPI_Wtime();

    if (rank == RANK_ROOT) {
        A = (double *)malloc(sizeof(double) * N * N);
        Ax = (double *)malloc(sizeof(double) * N);
        b = (double *)malloc(sizeof(double) * N);
        init_matrix(A, b, N);
        b_norm = vector_norm(b, N);
    }
    MPI_Bcast(&b_norm, 1, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    int n_partial = N / size;
    if (rank < N % size) {
        n_partial++;
    }
    double *a_partial = (double *)malloc(sizeof(double) * n_partial * N);
    distribute_matrix(A, N, a_partial, n_partial, rank, size);

    vector_mult_matrix(x, N, a_partial, n_partial, Ax, rank, size);
    double accuracy = BEGIN_ACCURACY;
    if (rank == RANK_ROOT) {
        vector_sub(Ax, b, Ax, N);
    }
    while (accuracy > EPSILON) {
        double iteration_start_time = MPI_Wtime();
        if (rank == RANK_ROOT) {
            vector_mult_scalar(Ax, TAU, Ax, N);
            vector_sub(x, Ax, x, N);
        }
        MPI_Bcast(x, (int)N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
        vector_mult_matrix(x, N, a_partial, n_partial, Ax, rank, size);
        if (rank == RANK_ROOT) {
            vector_sub(Ax, b, Ax, N);
            accuracy = vector_accuracy(Ax, b_norm, N);
            printf("iteration time = %.5f, accuracy = %.10f\n", MPI_Wtime() - iteration_start_time, accuracy);
        }
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    }

    if (rank == RANK_ROOT) {
        free(Ax);
        free(A);
        free(b);
        vector_print(x, N);

        printf("matrix size = %d\ntotal time = %.5f\n", (int)N, MPI_Wtime() - program_start_time);
    }
    free(a_partial);
    free(x);
    MPI_Finalize();
    return 0;
}
