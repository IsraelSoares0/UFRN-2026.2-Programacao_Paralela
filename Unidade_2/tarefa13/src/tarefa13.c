/*
 * Navier-Stokes 2D apenas com viscosidade (sem pressao e sem forcas externas)
 * TAREFA 13 (OpenMP): v4 (regiao paralela unica + kernel SIMD) com AFINIDADE DE THREADS
 *
 * Argumentos: <t_final> [N] [bind]   (N padrao 1024, bind padrao env)
 *
 * Compilar:  gcc -O3 -Wall -fopenmp -o tarefa13 tarefa13.c -lm -lmpascalops
 * Exemplos:  OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./tarefa13 0.001 2048
 *            OMP_NUM_THREADS=16 OMP_PLACES=cores ./tarefa13 0.001 2048 spread
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sched.h>
#include <omp.h>
#include "pascalops.h"

#define IDX(i, j, N) ((size_t)(i) * (N) + (j))

#define MODO   2
#define N_GRID 1024
#define NU     0.01

typedef enum { B_ENV, B_MASTER, B_CLOSE, B_SPREAD } Bind;

typedef struct {
    double *u, *v, *un, *vn;
    int     N, nsteps, saida;
    double  dt, r, dA;
    long    total;
    double  t_passo;
    int    *cpu_ini, *cpu_fim;   /* CPU de cada thread no inicio e no fim */
    long   *migr;                /* migracoes observadas por thread */
} Ctx;

static double g_um, g_vm, g_mu, g_mv, g_en;

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

static void integra(Ctx *c)
{
    const int N      = c->N;
    const int nsteps = c->nsteps;
    const long total = c->total;
    const double dt  = c->dt, dA = c->dA;

    const double *cu = c->u,  *cv = c->v;
    double       *nu_ = c->un, *nv_ = c->vn;
    const int tid = omp_get_thread_num();

    int  ult = sched_getcpu();
    long mig = 0;
    c->cpu_ini[tid] = ult;

    for (int n = 0; n <= nsteps; n++) {

        int cpu = sched_getcpu();
        if (cpu != ult) { mig++; ult = cpu; }

        if (n % c->saida == 0 || n == nsteps) {
            #pragma omp single
            { g_um = 0; g_vm = 0; g_mu = 0; g_mv = 0; g_en = 0; }

            #pragma omp for schedule(static) \
                    reduction(max:g_um, g_vm) reduction(+:g_mu, g_mv, g_en)
            for (long k = 0; k < total; k++) {
                double fu = fabs(cu[k]), fv = fabs(cv[k]);
                if (fu > g_um) g_um = fu;
                if (fv > g_vm) g_vm = fv;
                g_mu += cu[k];
                g_mv += cv[k];
                g_en += 0.5 * (cu[k] * cu[k] + cv[k] * cv[k]);
            }

            #pragma omp single
            printf("%8.4f %12.6e %12.6e %14.8e %14.8e %14.8e\n",
                   n * dt, g_um, g_vm, g_mu * dA, g_mv * dA, g_en * dA);
        }
        if (n == nsteps) break;

        double t0 = 0.0;
        if (tid == 0) t0 = omp_get_wtime();

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++)
            calc_linha(cu, cv, nu_, nv_, N, i, c->r);

        if (tid == 0) c->t_passo += omp_get_wtime() - t0;

        const double *tu = cu; cu = nu_; nu_ = (double *)tu;
        const double *tv = cv; cv = nv_; nv_ = (double *)tv;
    }

    c->cpu_fim[tid] = ult;
    c->migr[tid]    = mig;
}

static const char *nome_politica(omp_proc_bind_t p)
{
    switch (p) {
        case omp_proc_bind_false:  return "false";
        case omp_proc_bind_true:   return "true";
        case omp_proc_bind_master: return "master";
        case omp_proc_bind_close:  return "close";
        case omp_proc_bind_spread: return "spread";
        default:                   return "?";
    }
}

static const char *env_ou(const char *nome)
{
    const char *s = getenv(nome);
    return s ? s : "(nao definido)";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <t_final> [N] [env|master|close|spread]\n", argv[0]);
        return 1;
    }
    const double tf = atof(argv[1]);
    if (tf <= 0.0) { fprintf(stderr, "t_final deve ser > 0\n"); return 1; }

    const int N = (argc >= 3) ? atoi(argv[2]) : N_GRID;
    if (N < 3) { fprintf(stderr, "N deve ser >= 3\n"); return 1; }

    const char *bind_arg = (argc >= 4) ? argv[3] : "env";
    Bind bind;
    if      (!strcmp(bind_arg, "env"))    bind = B_ENV;
    else if (!strcmp(bind_arg, "master")) bind = B_MASTER;
    else if (!strcmp(bind_arg, "close"))  bind = B_CLOSE;
    else if (!strcmp(bind_arg, "spread")) bind = B_SPREAD;
    else {
        fprintf(stderr, "bind invalido '%s'. Use env|master|close|spread "
                        "(false/true: OMP_PROC_BIND no ambiente)\n", bind_arg);
        return 1;
    }

    const double nu = NU;
    const int threads = omp_get_max_threads();

    const double L  = 1.0;
    const double dx = L / N;
    const double r  = 0.2;
    double dt       = r * dx * dx / nu;
    int    nsteps   = (int)ceil(tf / dt);
    dt = tf / nsteps;
    const double r_real = nu * dt / (dx * dx);

    const char *bind_efetivo = (bind == B_ENV) ? nome_politica(omp_get_proc_bind())
                                               : bind_arg;

    printf("Modo %d | N=%d | nu=%g | dx=%g | dt=%g | passos=%d | r=%.4f (limite 0.25)\n",
           MODO, N, nu, dx, dt, nsteps, r_real);
    printf("OpenMP: threads=%d | schedule=static | versao=tarefa13 (regiao unica)\n", threads);
    printf("AFINIDADE: OMP_PROC_BIND=%s | OMP_PLACES=%s | clausula=%s | efetivo=%s | lugares=%d\n",
           env_ou("OMP_PROC_BIND"), env_ou("OMP_PLACES"), bind_arg,
           bind_efetivo, omp_get_num_places());
    if (r_real > 0.25) { fprintf(stderr, "Instavel! r > 0.25\n"); return 1; }

    size_t bytes = (size_t)N * N * sizeof(double);
    double *u  = malloc(bytes), *v  = malloc(bytes);
    double *un = malloc(bytes), *vn = malloc(bytes);
    int    *cpu_ini = malloc(threads * sizeof(int));
    int    *cpu_fim = malloc(threads * sizeof(int));
    long   *migr    = calloc(threads, sizeof(long));
    if (!u || !v || !un || !vn || !cpu_ini || !cpu_fim || !migr) {
        fprintf(stderr, "Sem memoria\n"); return 1;
    }
    for (int t = 0; t < threads; t++) { cpu_ini[t] = -1; cpu_fim[t] = -1; }

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

    Ctx ctx = {
        .u = u, .v = v, .un = un, .vn = vn,
        .N = N, .nsteps = nsteps,
        .saida = (nsteps / 5 < 1) ? 1 : nsteps / 5,
        .dt = dt, .r = r_real, .dA = dx * dx,
        .total = (long)N * N, .t_passo = 0.0,
        .cpu_ini = cpu_ini, .cpu_fim = cpu_fim, .migr = migr
    };

    double t_total0 = omp_get_wtime();
    pascal_start(1);

    switch (bind) {
        case B_MASTER: {
            #pragma omp parallel proc_bind(master)
            integra(&ctx);
            break;
        }
        case B_CLOSE: {
            #pragma omp parallel proc_bind(close)
            integra(&ctx);
            break;
        }
        case B_SPREAD: {
            #pragma omp parallel proc_bind(spread)
            integra(&ctx);
            break;
        }
        default: {
            #pragma omp parallel
            integra(&ctx);
            break;
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

    long migr_total = 0;
    printf("\nMAPA thread:cpu_inicio->cpu_fim (migracoes)\n");
    for (int t = 0; t < threads; t++) {
        migr_total += migr[t];
        printf("  %d:%d->%d (%ld)%s", t, cpu_ini[t], cpu_fim[t], migr[t],
               (t % 4 == 3 || t == threads - 1) ? "\n" : "");
    }

    printf("\nTempo nos passos de calculo: %.4f s | tempo total do laco: %.4f s\n",
           ctx.t_passo, t_total);
    printf("RESULT,%d,%d,%s,%d,%.6f,%.6f,%.3e,%ld\n",
           N, threads, bind_efetivo, nsteps, ctx.t_passo, t_total, err_max, migr_total);

    free(u); free(v); free(un); free(vn);
    free(cpu_ini); free(cpu_fim); free(migr);
    return 0;
}