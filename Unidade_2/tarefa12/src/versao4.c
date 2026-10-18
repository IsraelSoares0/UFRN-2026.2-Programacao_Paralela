/*
 * Navier-Stokes 2D apenas com viscosidade (sem pressao e sem forcas externas)
 * VERSAO 4 (OpenMP): REGIAO PARALELA UNICA + kernel SIMD da v3
 *
 * Melhoria em relacao a v3:
 *   - Nas versoes anteriores, cada passo de tempo abria e fechava uma
 *     regiao paralela (fork/join) e os diagnosticos abriam outra. Com
 *     muitas threads, o trabalho por passo por thread fica pequeno e esse
 *     custo passa a dominar (limita a escalabilidade forte).
 *   - Aqui a regiao paralela e criada UMA vez, fora do laco temporal.
 *     Dentro dela, cada passo e um "omp for" com barreira implicita
 *     (necessaria: o passo n+1 depende do passo n). A troca dos buffers
 *     (u <-> un) e feita em ponteiros privados de cada thread.
 *
 * Observacao sobre o pascalops: pascal_start/stop(1) cobrem toda a regiao
 * paralela (todos os passos + diagnosticos), nao um passo isolado.
 *
 * Argumentos: <t_final> [N]   (N opcional, padrao 1024)
 *
 * Compilar:  gcc -O3 -Wall -fopenmp -o ns_v4 ns_visc_v4.c -lm -lmpascalops
 * Uso:       ./ns_v4 <t_final> [N]
 * Exemplo:   OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ./ns_v4 0.01
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "pascalops.h"

#define IDX(i, j, N) ((size_t)(i) * (N) + (j))

#define MODO   2
#define N_GRID 1024
#define NU     0.01

/* Mesmo kernel da v3: miolo vetorizavel + duas bordas periodicas */
static inline void calc_linha(const double *restrict u, const double *restrict v,
                              double *restrict un, double *restrict vn,
                              int N, int i, double r)
{
    const int im = (i == 0)     ? N - 1 : i - 1;
    const int ip = (i == N - 1) ? 0     : i + 1;
    const size_t c = (size_t)i  * N;
    const size_t a = (size_t)im * N;
    const size_t b = (size_t)ip * N;

    #pragma omp simd
    for (int j = 1; j < N - 1; j++) {
        double lap_u = u[b + j] + u[a + j] + u[c + j + 1] + u[c + j - 1]
                     - 4.0 * u[c + j];
        double lap_v = v[b + j] + v[a + j] + v[c + j + 1] + v[c + j - 1]
                     - 4.0 * v[c + j];
        un[c + j] = u[c + j] + r * lap_u;
        vn[c + j] = v[c + j] + r * lap_v;
    }

    const int bordas[2] = {0, N - 1};
    for (int k = 0; k < 2; k++) {
        const int j  = bordas[k];
        const int jm = (j == 0)     ? N - 1 : j - 1;
        const int jp = (j == N - 1) ? 0     : j + 1;
        double lap_u = u[b + j] + u[a + j] + u[c + jp] + u[c + jm]
                     - 4.0 * u[c + j];
        double lap_v = v[b + j] + v[a + j] + v[c + jp] + v[c + jm]
                     - 4.0 * v[c + j];
        un[c + j] = u[c + j] + r * lap_u;
        vn[c + j] = v[c + j] + r * lap_v;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <t_final> [N]\n", argv[0]);
        return 1;
    }
    const double tf = atof(argv[1]);
    if (tf <= 0.0) { fprintf(stderr, "t_final deve ser > 0\n"); return 1; }

    const int N = (argc >= 3) ? atoi(argv[2]) : N_GRID;
    if (N < 3) { fprintf(stderr, "N deve ser >= 3\n"); return 1; }
    const double nu = NU;
    const int threads = omp_get_max_threads();

    const double L  = 1.0;
    const double dx = L / N;
    const double r  = 0.2;
    double dt       = r * dx * dx / nu;
    int    nsteps   = (int)ceil(tf / dt);
    dt = tf / nsteps;
    const double r_real = nu * dt / (dx * dx);

    printf("Modo %d | N=%d | nu=%g | dx=%g | dt=%g | passos=%d | r=%.4f (limite 0.25)\n",
           MODO, N, nu, dx, dt, nsteps, r_real);
    printf("OpenMP: threads=%d | schedule=static | chunk=0 | collapse=1 | versao=v4 (regiao unica)\n",
           threads);
    if (r_real > 0.25) { fprintf(stderr, "Instavel! r > 0.25\n"); return 1; }

    size_t bytes = (size_t)N * N * sizeof(double);
    double *u  = malloc(bytes), *v  = malloc(bytes);
    double *un = malloc(bytes), *vn = malloc(bytes);
    if (!u || !v || !un || !vn) { fprintf(stderr, "Sem memoria\n"); return 1; }

    const double A = 1.0, s0 = 0.05;
    const double xc = 0.5 * L, yc = 0.5 * L;

    /* Inicializacao paralela */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
            double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
            u[IDX(i, j, N)]  = A * exp(-r2 / (2 * s0 * s0));
            v[IDX(i, j, N)]  = 0.0;
            un[IDX(i, j, N)] = 0.0;
            vn[IDX(i, j, N)] = 0.0;
        }
    }

    printf("%8s %12s %12s %14s %14s %14s\n",
           "t", "max|u|", "max|v|", "mom_u", "mom_v", "energia");

    /* Variaveis compartilhadas dos diagnosticos e do tempo de calculo */
    double um = 0, vm = 0, mu = 0, mv = 0, en = 0;
    const double dA = dx * dx;
    const long total = (long)N * N;
    double t_passo = 0.0;
    const int saida = (nsteps / 5 < 1) ? 1 : nsteps / 5;

    double t_total0 = omp_get_wtime();
    pascal_start(1);

    #pragma omp parallel
    {
        const double *cu = u,  *cv = v;
        double       *nu_ = un, *nv_ = vn;
        const int tid = omp_get_thread_num();

        for (int n = 0; n <= nsteps; n++) {

            if (n % saida == 0 || n == nsteps) {
                #pragma omp single
                { um = 0; vm = 0; mu = 0; mv = 0; en = 0; }

                #pragma omp for schedule(static) \
                        reduction(max:um, vm) reduction(+:mu, mv, en)
                for (long k = 0; k < total; k++) {
                    double fu = fabs(cu[k]), fv = fabs(cv[k]);
                    if (fu > um) um = fu;
                    if (fv > vm) vm = fv;
                    mu += cu[k];
                    mv += cv[k];
                    en += 0.5 * (cu[k] * cu[k] + cv[k] * cv[k]);
                }

                #pragma omp single
                printf("%8.4f %12.6e %12.6e %14.8e %14.8e %14.8e\n",
                       n * dt, um, vm, mu * dA, mv * dA, en * dA);
            }
            if (n == nsteps) break;

            double t0 = 0.0;
            if (tid == 0) t0 = omp_get_wtime();

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++)
                calc_linha(cu, cv, nu_, nv_, N, i, r_real);

            if (tid == 0) t_passo += omp_get_wtime() - t0;

            const double *tu = cu; cu = nu_; nu_ = (double *)tu;
            const double *tv = cv; cv = nv_; nv_ = (double *)tv;
        }
    }

    pascal_stop(1);
    double t_total = omp_get_wtime() - t_total0;

    const double *uf = (nsteps % 2) ? un : u;

    /* Validacao contra a solucao analitica */
    double s2 = s0 * s0 + 2 * nu * tf;
    double err_max = 0, err_l2 = 0, ref_l2 = 0;
    #pragma omp parallel for schedule(static) \
            reduction(max:err_max) reduction(+:err_l2, ref_l2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
            double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
            double ex = A * (s0 * s0 / s2) * exp(-r2 / (2 * s2));
            double d = uf[IDX(i, j, N)] - ex;
            if (fabs(d) > err_max) err_max = fabs(d);
            err_l2 += d * d;
            ref_l2 += ex * ex;
        }
    }
    printf("\nComparacao com solucao analitica em t=%g:\n", tf);
    printf("  erro maximo      = %.3e\n", err_max);
    printf("  erro L2 relativo = %.3e\n", sqrt(err_l2 / ref_l2));

    printf("\nTempo nos passos de calculo: %.4f s | tempo total do laco: %.4f s\n",
           t_passo, t_total);
    printf("RESULT,%d,%d,%s,%d,%d,%.6f,%.6f,%.3e\n",
           N, threads, "static", 0, nsteps, t_passo, t_total, err_max);

    free(u); free(v); free(un); free(vn);
    return 0;
}