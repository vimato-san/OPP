#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>

#define SIZE 8000
#define EPSILON 0.0001
#define TAU 0.0001

void vector_print(double *vector, size_t vector_size) {
    printf("{");
    for (int i = 0; i < vector_size; i++)
        printf("%.3f, ", vector[i]);
    printf("\b\b}\n");
}


double vector_norm(const double *vector, size_t vector_size) {
    double sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < vector_size; i++) {
        #pragma omp atomic
        sum += (vector[i] * vector[i]);
    }
    return sqrt(sum);
}


void vector_sub(const double *vec_1, const double *vec_2, double *dest, size_t size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
        dest[i] = vec_1[i] - vec_2[i];
}


void vector_mult_scalar(const double *vector, double scalar, double *dest, size_t size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
        dest[i] = vector[i] * scalar;
}


void vector_mult_matrix(const double *vector, const double *matrix, double *result, size_t N) {
    omp_set_num_threads(12);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            double sum = 0;
            for (int j = 0; j < N; j++)
                sum += matrix[i * N + j] * vector[j];
            result[i] = sum;
        }
    }
}


void init_matrices(double *A, double *b, double *x, size_t N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            if (i == j)
                A[N * i + j] = 2;
            else
                A[N * i + j] = 1;
        b[i] = (int)N + 1;
        x[i] = 0;
    }
}


double vector_accuracy(double *Ax, double *b, double b_norm, size_t N) {
    double *tmp = (double *)malloc(sizeof(double) * N);
    vector_sub(Ax, b, tmp, N);
    double res = vector_norm(tmp, N) / b_norm;

    free(tmp);
    return res;
}


int main(int argc, char *argv[]) {
    size_t N = SIZE;
    double *A = (double *)malloc(sizeof(double) * N * N);
    double *x = (double *)malloc(sizeof(double) * N);
    double *Ax = (double *)malloc(sizeof(double) * N);
    double *b = (double *)malloc(sizeof(double) * N);

    init_matrices(A, b, x, N);
    double time_begin = omp_get_wtime();
    double b_norm = vector_norm(b, N);
    vector_mult_matrix(x, A, Ax, N);

    double accuracy = 1;
    while (accuracy > EPSILON) {
        vector_sub(Ax, b, Ax, N);
        vector_mult_scalar(Ax, TAU, Ax, N);
        vector_sub(x, Ax, x, N);
        vector_mult_matrix(x, A, Ax, N);
        accuracy = vector_accuracy(Ax, b, b_norm, N);
        printf("%.10f\n", accuracy);
    }
    vector_print(x, N);
    double time_end = omp_get_wtime();
    printf("Time taken: %lf\n", time_end - time_begin);
    return 0;
}
