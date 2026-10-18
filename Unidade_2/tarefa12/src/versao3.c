/*
 * Navier-Stokes 2D apenas com viscosidade (sem pressao e sem forcas externas)
 * VERSAO 3 (OpenMP): LACO REESTRUTURADO + SIMD + schedule(static)
 *
 * Melhorias em relacao as versoes schedule / collapse:
 *   1) O miolo do laco interno (j = 1 .. N-2) nao tem mais condicionais
 *      de contorno; as bordas j = 0 e j = N-1 sao tratadas a parte.
 *      Isso permite vetorizar (#pragma omp simd).
 *   2) schedule(static) fixo: o custo por linha e uniforme, entao static
 *      tem overhead minimo e preserva a localidade NUMA.
 *   3) Ponteiros de linha em vez de IDX() repetido (menos aritmetica).
 *
 * Argumentos: <t_final> [N]   (N opcional, padrao 1024)
 *   O N opcional permite variar o tamanho do problema para a analise de
 *   escalabilidade fraca.
 *
 * Compilar:  gcc -O3 -Wall -fopenmp -o ns_v3 ns_visc_v3.c -lm -lmpascalops
 * Uso:       ./ns_v3 <t_final> [N]
 * Exemplo:   OMP_NUM_THREADS=8 ./ns_v3 0.01
 *            OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ./ns_v3 0.01 2048
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

/* Atualiza uma linha i (u e v). Miolo vetorizavel + duas bordas periodicas. */
static inline void calc_linha(const double *restrict u, const double *restrict v,
                              double *restrict un, double *restrict vn,
                              int N, int i, double r)
{
    const int im = (i == 0)     ? N - 1 : i - 1;
    const int ip = (i == N - 1) ? 0     : i + 1;
    const size_t c = (size_t)i  * N;
    const size_t a = (size_t)im * N;
    const size_t b = (size_t)ip * N;

    /* sem condicionais -> o compilador vetoriza */
    #pragma omp simd
    for (int j = 1; j < N - 1; j++) {
        double lap_u = u[b + j] + u[a + j] + u[c + j + 1] + u[c + j - 1]
                     - 4.0 * u[c + j];
        double lap_v = v[b + j] + v[a + j] + v[c + j + 1] + v[c + j - 1]
                     - 4.0 * v[c + j];
        un[c + j] = u[c + j] + r * lap_u;
        vn[c + j] = v[c + j] + r * lap_v;
    }

    /* bordas j = 0 e j = N-1 */
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

static void passo(const double *u, const double *v,
                  double *un, double *vn, int N, double r)
{
    pascal_start(1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        calc_linha(u, v, un, vn, N, i, r);
    pascal_stop(1);
}

/* Diagnosticos: max|u|, max|v|, momento total e energia cinetica */
static void diagnosticos(const double *u, const double *v, int N, double dx,
                         double *umax, double *vmax, double *mom_u,
                         double *mom_v, double *energia)
{
    double a = 0, b = 0, su = 0, sv = 0, e = 0;
    const long total = (long)N * N;

    #pragma omp parallel for schedule(static) \
            reduction(max:a, b) reduction(+:su, sv, e)
    for (long k = 0; k < total; k++) {
        double fu = fabs(u[k]), fv = fabs(v[k]);
        if (fu > a) a = fu;
        if (fv > b) b = fv;
        su += u[k];
        sv += v[k];
        e  += 0.5 * (u[k] * u[k] + v[k] * v[k]);
    }
    double dA = dx * dx;
    *umax = a; *vmax = b;
    *mom_u = su * dA; *mom_v = sv * dA; *energia = e * dA;
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
    printf("OpenMP: threads=%d | schedule=static | chunk=0 | collapse=1 | versao=v3 (simd)\n",
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

    double t_passo  = 0.0;
    double t_total0 = omp_get_wtime();

    int saida = nsteps / 5; if (saida < 1) saida = 1;
    for (int n = 0; n <= nsteps; n++) {
        if (n % saida == 0 || n == nsteps) {
            double um, vm, mu, mv, e;
            diagnosticos(u, v, N, dx, &um, &vm, &mu, &mv, &e);
            printf("%8.4f %12.6e %12.6e %14.8e %14.8e %14.8e\n",
                   n * dt, um, vm, mu, mv, e);
        }
        if (n == nsteps) break;

        double t0 = omp_get_wtime();
        passo(u, v, un, vn, N, r_real);
        t_passo += omp_get_wtime() - t0;

        double *t;
        t = u; u = un; un = t;
        t = v; v = vn; vn = t;
    }
    double t_total = omp_get_wtime() - t_total0;

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
            double d = u[IDX(i, j, N)] - ex;
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