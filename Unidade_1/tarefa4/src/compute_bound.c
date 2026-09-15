/*
 * compute_bound.c
 * -----------------------------------------------------------------------
 * Benchmark COMPUTE-BOUND (CPU-bound): muitas operações matemáticas
 * (sin, cos, sqrt, potências) por elemento de um vetor pequeno/médio.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char **argv) {
    long n = 2 * 1000 * 1000L;  /* vetor bem menor */
    int threads = 1;
    int inner_iters = 400;      /* trabalho de CPU por elemento */
    int reps = 5;

    if (argc > 1) threads = atoi(argv[1]);
    if (argc > 2) n = atol(argv[2]);
    if (argc > 3) inner_iters = atoi(argv[3]);
    if (argc > 4) reps = atoi(argv[4]);

    omp_set_num_threads(threads);

    double *a = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));
    if (!a || !out) {
        fprintf(stderr, "Falha ao alocar memoria para n=%ld\n", n);
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        a[i] = 0.0001 + (double)(i % 9973) * 0.00001;
    }

    double best = 1e18, total = 0.0;

    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();

        /* ---- Kernel compute-bound ---- */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) {
            double x = a[i];
            double acc = x;
            for (int k = 0; k < inner_iters; k++) {
                acc = sin(acc) * cos(acc) + sqrt(fabs(acc)) * 0.5;
                acc = acc * 1.0000001 + 1e-9;
            }
            out[i] = acc;
        }
        /* ------------------------------- */

        double t1 = omp_get_wtime();
        double dt = t1 - t0;
        total += dt;
        if (dt < best) best = dt;
    }

    volatile double sink = out[0] + out[n / 2] + out[n - 1];
    (void) sink;

    printf("%-10s %-12s %-15s %-12s %-10s\n",
       "Threads", "Melhor", "Pontos", "Iter", "Reps");

    printf("---------------------------------------------------------------\n");

    printf("%-10d %-12.6f %-15ld %-12d %-10d\n",
        threads, best, n, inner_iters, reps);

    free(a); free(out);
    return 0;
}