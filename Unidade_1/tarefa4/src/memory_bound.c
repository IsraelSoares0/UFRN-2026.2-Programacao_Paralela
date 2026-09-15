/*
 * memory_bound.c
 * -----------------------------------------------------------------------
 * Benchmark MEMORY-BOUND: soma simples de vetores (c[i] = a[i] + b[i]).
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    long n = 60 * 1000 * 1000L; /* 60 milhões de doubles => ~480MB por vetor */
    int threads = 1;
    int reps = 5;

    if (argc > 1) threads = atoi(argv[1]);
    if (argc > 2) n = atol(argv[2]);
    if (argc > 3) reps = atoi(argv[3]);

    omp_set_num_threads(threads);

    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "Falha ao alocar memoria para n=%ld\n", n);
        return 1;
    }

    /* Inicialização paralela */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        a[i] = (double)(i % 1000) * 0.5;
        b[i] = (double)(i % 777) * 1.3;
    }

    double best = 1e18, total = 0.0;

    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();

        /* ---- Kernel memory-bound ---- */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
        /* ------------------------------ */

        double t1 = omp_get_wtime();
        double dt = t1 - t0;
        total += dt;
        if (dt < best) best = dt;
    }

    /* Evita que o compilador elimine o cálculo por "dead code elimination" */
    volatile double sink = c[0] + c[n / 2] + c[n - 1];
    (void) sink;

    printf("%-10s %-12s %-15s %-10s\n",
       "Threads", "Melhor", "Pontos", "Reps");

    printf("---------------------------------------------------------------\n");

    printf("%-10d %-12.6f %-15ld %-10d\n",
        threads, best, n, reps);

    free(a); free(b); free(c);
    return 0;
}