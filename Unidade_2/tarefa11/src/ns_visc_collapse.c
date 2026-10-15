/*
 * Navier-Stokes 2D apenas com viscosidade (sem pressao e sem forcas externas)
 * VERSAO OpenMP: COLLAPSE  (collapse(2) nos lacos i e j)
 *
 *     du/dt = nu * (d2u/dx2 + d2u/dy2)
 *     dv/dt = nu * (d2v/dx2 + d2v/dy2)
 *
 * Discretizacao: diferencas finitas, esquema explicito FTCS, 5 pontos.
 * Contorno: periodico nas duas direcoes.
 * Condicao inicial: gaussiana em u (modo 2), com solucao analitica.
 *
 * Parametros fixos no codigo: MODO = 2, N = 1024, nu = 0.01
 * Unico argumento do usuario: tempo final (t_final)
 *
 * Aqui os lacos i e j sao fundidos com collapse(2), usando
 * schedule(static) fixo. O numero de threads vem de OMP_NUM_THREADS
 * (o pascalanalyzer ajusta essa variavel a partir da opcao -c).
 *
 * Compilar:  gcc -O2 -Wall -fopenmp -o ns_collapse ns_visc_collapse.c -lm -lmpascalops
 *
 * Uso:       ./ns_collapse <t_final>
 * Exemplo:   ./ns_collapse 0.01
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "pascalops.h"

#define IDX(i, j, N) ((i) * (N) + (j))

#define MODO   2
#define N_GRID 1024
#define NU     0.01

static void passo(const double *u, const double *v,
                  double *un, double *vn, int N, double r)
{
    pascal_start(1);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int im = (i == 0)     ? N - 1 : i - 1;
            int ip = (i == N - 1) ? 0     : i + 1;
            int jm = (j == 0)     ? N - 1 : j - 1;
            int jp = (j == N - 1) ? 0     : j + 1;

            double lap_u = u[IDX(ip, j, N)] + u[IDX(im, j, N)]
                         + u[IDX(i, jp, N)] + u[IDX(i, jm, N)]
                         - 4.0 * u[IDX(i, j, N)];
            double lap_v = v[IDX(ip, j, N)] + v[IDX(im, j, N)]
                         + v[IDX(i, jp, N)] + v[IDX(i, jm, N)]
                         - 4.0 * v[IDX(i, j, N)];

            un[IDX(i, j, N)] = u[IDX(i, j, N)] + r * lap_u;
            vn[IDX(i, j, N)] = v[IDX(i, j, N)] + r * lap_v;
        }
    }
    pascal_stop(1);
}

/* Diagnosticos: max|u|, max|v|, momento total e energia cinetica */
static void diagnosticos(const double *u, const double *v, int N, double dx,
                         double *umax, double *vmax, double *mom_u,
                         double *mom_v, double *energia)
{
    double a = 0, b = 0, su = 0, sv = 0, e = 0;
    const int total = N * N;

    #pragma omp parallel for schedule(static) \
            reduction(max:a, b) reduction(+:su, sv, e)
    for (int k = 0; k < total; k++) {
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

static const char *nome_schedule(omp_sched_t k) {
    switch ((int)k & 0x7fffffff) {
        case omp_sched_static:  return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided:  return "guided";
        case omp_sched_auto:    return "auto";
        default:                return "outro";
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <t_final>\n", argv[0]);
        return 1;
    }
    const double tf = atof(argv[1]);
    if (tf <= 0.0) { fprintf(stderr, "t_final deve ser > 0\n"); return 1; }

    const int    N  = N_GRID;
    const double nu = NU;

    const omp_sched_t kind_atual = omp_sched_static;  /* fixo no pragma */
    const int chunk_atual = 0;
    const int threads = omp_get_max_threads();

    const double L  = 1.0;
    const double dx = L / N;
    const double r  = 0.2;                 /* nu*dt/dx^2 (<= 0.25) */
    double dt       = r * dx * dx / nu;
    int    nsteps   = (int)ceil(tf / dt);
    dt = tf / nsteps;
    const double r_real = nu * dt / (dx * dx);

    printf("Modo %d | N=%d | nu=%g | dx=%g | dt=%g | passos=%d | r=%.4f (limite 0.25)\n",
           MODO, N, nu, dx, dt, nsteps, r_real);
    printf("OpenMP: threads=%d | schedule=%s | chunk=%d | collapse=2\n",
           threads, nome_schedule(kind_atual), chunk_atual);
    if (r_real > 0.25) { fprintf(stderr, "Instavel! r > 0.25\n"); return 1; }

    size_t bytes = (size_t)N * N * sizeof(double);
    double *u  = malloc(bytes), *v  = malloc(bytes);
    double *un = malloc(bytes), *vn = malloc(bytes);
    if (!u || !v || !un || !vn) { fprintf(stderr, "Sem memoria\n"); return 1; }

    const double A = 1.0, s0 = 0.05;
    const double xc = 0.5 * L, yc = 0.5 * L;

    /* Inicializacao paralela (gaussiana em u, v = 0) */
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
           N, threads, nome_schedule(kind_atual), chunk_atual,
           nsteps, t_passo, t_total, err_max);

    free(u); free(v); free(un); free(vn);
    return 0;
}
