/*
 * Navier-Stokes 2D apenas com viscosidade (sem pressão e sem forças externas)
 * VERSÃO PARALELA COM OpenMP
 *
 *     du/dt = nu * (d2u/dx2 + d2u/dy2)
 *     dv/dt = nu * (d2v/dx2 + d2v/dy2)
 *
 * Discretização: diferenças finitas, esquema explícito FTCS, 5 pontos.
 * Contorno: periódico nas duas direções.
 * Estabilidade (2D): r = nu*dt/dx^2 <= 0.25
 *
 * Compilar:  gcc -O2 -Wall -fopenmp -o ns_visc_omp ns_visc_omp.c -lm
 *
 * Uso:
 *   ./ns_visc_omp <modo> [N] [t_final] [nu] [quadros] [threads] [schedule] [chunk] [collapse]
 *
 *   modo      0 = parado | 1 = velocidade constante | 2 = gaussiana (analítica)
 *   quadros   > 1 grava snapshots.bin (0 = não grava)
 *   threads   número de threads (0 = padrão do OpenMP / OMP_NUM_THREADS)
 *   schedule  static | dynamic | guided | auto  (omitido = usa OMP_SCHEDULE)
 *   chunk     tamanho do chunk (0 = padrão da política)
 *   collapse  0 = paralelo só no laço i | 1 = collapse(2) nos laços i e j
 *
 * Exemplo:  ./ns_visc_omp 2 1024 0.01 0.01 0 8 dynamic 16 1
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define IDX(i, j, N) ((i) * (N) + (j))

/**
 * VARIANTE A: paraleliza apenas o laço externo (i)
 */
static void passo_externo(const double *u, const double *v,
                          double *un, double *vn, int N, double r)
{
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        int im = (i == 0)     ? N - 1 : i - 1;
        int ip = (i == N - 1) ? 0     : i + 1;
        for (int j = 0; j < N; j++) {
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
}

/**
 * VARIANTE B: collapse(2) nos laços i e j
 * para utilizar collapse, im/ip foram movidos para dentro do corpo
 * recalculados a cada ponto
 */
static void passo_collapse(const double *u, const double *v,
                           double *un, double *vn, int N, double r)
{
    #pragma omp parallel for collapse(2) schedule(runtime)
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
}

/* Diagnósticos: max|u|, max|v|, momento total e energia cinética */
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

static int parse_schedule(const char *s, omp_sched_t *kind) {
    if (!strcmp(s, "static"))  { *kind = omp_sched_static;  return 1; }
    if (!strcmp(s, "dynamic")) { *kind = omp_sched_dynamic; return 1; }
    if (!strcmp(s, "guided"))  { *kind = omp_sched_guided;  return 1; }
    if (!strcmp(s, "auto"))    { *kind = omp_sched_auto;    return 1; }
    
    return 0;
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
    int    modo = (argc > 1) ? atoi(argv[1]) : 2;
    int    N    = (argc > 2) ? atoi(argv[2]) : 128;
    double tf   = (argc > 3) ? atof(argv[3]) : 0.5;
    double nu   = (argc > 4) ? atof(argv[4]) : 0.01;
    int    nframes  = (argc > 5) ? atoi(argv[5]) : 0;
    int    nthreads = (argc > 6) ? atoi(argv[6]) : 0;
    int    chunk    = (argc > 8) ? atoi(argv[8]) : 0;
    int    collapse = (argc > 9) ? atoi(argv[9]) : 0;
 
    /* --- configuração do OpenMP --- */
    if (nthreads > 0) omp_set_num_threads(nthreads);
    if (argc > 7) {
        omp_sched_t kind;
        if (!parse_schedule(argv[7], &kind)) {
            fprintf(stderr, "schedule invalido: %s (static|dynamic|guided|auto)\n", argv[7]);
            return 1;
        }
        omp_set_schedule(kind, chunk);     /* lido por schedule(runtime) */
    }
    omp_sched_t kind_atual; int chunk_atual;
    omp_get_schedule(&kind_atual, &chunk_atual);
    int threads_efetivas = omp_get_max_threads();
 
    void (*passo)(const double *, const double *, double *, double *, int, double) = collapse ? passo_collapse : passo_externo;
 
    const double L  = 1.0;
    const double dx = L / N;
    const double r  = 0.2;                 /* nu*dt/dx^2 (<= 0.25) */
    double dt       = r * dx * dx / nu;
    int    nsteps   = (int)ceil(tf / dt);
    dt = tf / nsteps;
    double r_real   = nu * dt / (dx * dx);
 
    printf("Modo %d | N=%d | nu=%g | dx=%g | dt=%g | passos=%d | r=%.4f (limite 0.25)\n",
           modo, N, nu, dx, dt, nsteps, r_real);
    printf("OpenMP: threads=%d | schedule=%s | chunk=%d | collapse=%d\n",
           threads_efetivas, nome_schedule(kind_atual), chunk_atual, collapse ? 2 : 1);
    if (r_real > 0.25) { fprintf(stderr, "Instavel! r > 0.25\n"); return 1; }
 
    size_t bytes = (size_t)N * N * sizeof(double);
    double *u  = malloc(bytes), *v  = malloc(bytes);
    double *un = malloc(bytes), *vn = malloc(bytes);
    if (!u || !v || !un || !vn) { fprintf(stderr, "Sem memoria\n"); return 1; }
 
    const double U0 = 1.0, V0 = 0.5;       /* modo 1 */
    const double A = 1.0, s0 = 0.05;       /* modo 2 */
    const double xc = 0.5 * L, yc = 0.5 * L;
 
    /**
     * INICILIZAÇÃO PARALELA
     */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
            double uu = 0, vv = 0;
            if (modo == 1) { uu = U0; vv = V0; }
            if (modo == 2) {
                double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
                uu = A * exp(-r2 / (2 * s0 * s0));
            }
            u[IDX(i, j, N)]  = uu;
            v[IDX(i, j, N)]  = vv;
            un[IDX(i, j, N)] = 0.0;
            vn[IDX(i, j, N)] = 0.0;
        }
    }
 
    double umax0, vmax0, mu0, mv0, e0;
    diagnosticos(u, v, N, dx, &umax0, &vmax0, &mu0, &mv0, &e0);
    printf("%8s %12s %12s %14s %14s %14s\n",
           "t", "max|u|", "max|v|", "mom_u", "mom_v", "energia");
 
    /* Snapshots*/
    FILE *fsnap   = NULL;
    int   proximo = 0;
    if (nframes > 1) {
        fsnap = fopen("snapshots.bin", "wb");
        if (!fsnap) { fprintf(stderr, "Nao abriu snapshots.bin\n"); return 1; }
        int32_t hdr[2] = { N, nframes };
        double  par[4] = { nu, A, s0, L };
        fwrite(hdr, sizeof hdr, 1, fsnap);
        fwrite(par, sizeof par, 1, fsnap);
    }
 
    double t_passo = 0.0;
    double t_total0 = omp_get_wtime();
 
    int saida = nsteps / 5; if (saida < 1) saida = 1;
    for (int n = 0; n <= nsteps; n++) {
        if (fsnap && proximo < nframes &&
            n == (int)llround((double)proximo * nsteps / (nframes - 1))) {
            double t_now = n * dt;
            fwrite(&t_now, sizeof t_now, 1, fsnap);
            fwrite(u, sizeof(double), (size_t)N * N, fsnap);
            fwrite(v, sizeof(double), (size_t)N * N, fsnap);
            proximo++;
        }
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
 
    /* Validação do modo 2 contra solução analítica */
    double err_max = 0;
    if (modo == 2) {
        double s2 = s0 * s0 + 2 * nu * tf;
        double err_l2 = 0, ref_l2 = 0;
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
    }
 
    if (modo == 0 || modo == 1) {
        double dev = 0;
        for (int k = 0; k < N * N; k++) {
            double eu = (modo == 1) ? U0 : 0, ev = (modo == 1) ? V0 : 0;
            dev = fmax(dev, fmax(fabs(u[k] - eu), fabs(v[k] - ev)));
        }
        printf("\nDesvio maximo em relacao ao estado inicial: %.3e %s\n", dev,
               dev < 1e-12 ? "(OK: campo estavel)" : "(ATENCAO)");
    }
 
    if (fsnap) { fclose(fsnap); printf("\n%d quadros gravados em snapshots.bin\n", proximo); }
 
    /* Linha de resultado em formato CSV, fácil de coletar por script */
    printf("\nTempo nos passos de calculo: %.4f s | tempo total do laco: %.4f s\n",
           t_passo, t_total);
    printf("RESULT,%d,%d,%d,%s,%d,%d,%d,%.6f,%.6f,%.3e\n",
           modo, N, threads_efetivas, nome_schedule(kind_atual), chunk_atual,
           collapse ? 2 : 1, nsteps, t_passo, t_total, err_max);
 
    free(u); free(v); free(un); free(vn);
    return 0;
}
