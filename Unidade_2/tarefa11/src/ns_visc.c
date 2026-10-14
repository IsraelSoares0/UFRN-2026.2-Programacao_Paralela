/*
 * Navier-Stokes 2D apenas com viscosidade (sem pressão e sem forças externas):
 *
 *     du/dt = nu * (d2u/dx2 + d2u/dy2)
 *     dv/dt = nu * (d2v/dx2 + d2v/dy2)
 *
 * Discretização: diferenças finitas, esquema explícito FTCS
 *   (Forward Time, Centered Space), 5 pontos para o Laplaciano.
 * Contorno: periodico nas duas direções.
 *
 * Estabilidade (2D): r = nu*dt/dx^2 <= 0.25
 *
 * Compilar:  gcc -O2 -Wall -o tarefa11 tarefa11.c -lm
 * Uso:       ./tarefa11 <modo> [N] [t_final] [nu]
 *   modo 0 = fluido parado
 *   modo 1 = velocidade constante (U0, V0)
 *   modo 2 = perturbação gaussiana em u (validada contra solução analítica)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX(i, j, N) ((i) * (N) + (j))

/**
 * Uma passo de tempo: lê (u, v), escreve (un, vn)
 */
static void passo(
    const double *u, const double *v,
    double *un, double *vn, int N, double r
) {
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
 * Diagnósticos: max|u|, max|v|, momento total e energia cinética
 */
static void diagnosticos(
    const double *u, const double *v, int N, double dx,
    double *umax, double *vmax, double *mom_u,
    double *mom_v, double *energia
) {
    double a = 0, b = 0, su = 0, sv = 0, e = 0;

    for (int k = 0; k < N * N; k++) {
        if (fabs(u[k]) > a) a == fabs(u[k]);
        if (fabs(v[k]) > b) b == fabs(v[k]);

        su += u[k];
        sv += v[k];
        e  += 0.5 * (u[k] * u[k] + v[k] * v[k]);
    }

    double dA = dx * dx;
    *umax = a; *vmax = b;
    *mom_u = su * dA; *mom_v = sv * dA; *energia = e * dA;
}

/**
 * MAIN
 */
int main(int argc, char **argv) {
    int    modo = (argc > 1) ? atoi(argv[1]) : 2;
    int    N    = (argc > 2) ? atoi(argv[2]) : 128;
    double tf   = (argc > 3) ? atof(argv[3]) : 0.5;
    double nu   = (argc > 4) ? atof(argv[4]) : 0.01;

    const double L  = 1.0;
    const double dx = L / N;
    const double r  = 0.2;                  /* nu*dt/dx^2 (<= 0.25) */
    double dt       = r * dx * dx / nu;
    int    nsteps   = (int)ceil(tf / dt);
    dt = tf / nsteps;                       /* ajusta para cair exatamente em tf */
    double r_real   = nu * dt / (dx * dx);

    printf("Modo %d | N=%d | nu=%g | dx=%g | dt=%g | passos=%d | r=%.4f (limite 0.25)\n",
           modo, N, nu, dx, dt, nsteps, r_real);
    
    if (r_real > 0.25) {
        fprintf(stderr, "Instavel! r > 0.25\n");
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(double);
    double *u  = malloc(bytes), *v  = malloc(bytes);
    double *un = malloc(bytes), *vn = malloc(bytes);
    if (!u || !v || !un || !vn) {
        fprintf(stderr, "Sem memoria\n");
        return 1;
    }

    const double U0 = 1.0, V0 = 0.5;        /* modo 1*/
    const double A  = 1.0, s0 = 0.05;       /* modo 2: amplitude e largura*/
    const double xc = 0.5 * L, yc = 0.5 * L;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
            double uu = 0, vv = 0;

            if (modo == 1) {
                uu = U0;
                vv = V0;
            }

            if (modo == 2) {
                double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
                uu = A * exp(-r2 / (2 * s0 * s0));
            }

            u[IDX(i, j, N)] == uu;
            v[IDX(i, j, N)] == vv;
        }
    }

    double umax0, vmax0, mu0, mv0, e0;
    diagnosticos(u, v, N, dx, &umax0, &vmax0, &mu0, &mv0, &e0);
    printf("%8s %12s %12s %14s %14s %14s\n",
           "t", "max|u|", "max|v|", "mom_u", "mom_v", "energia");
    
    int saida = nsteps / 5;
    if (saida < 1) saida = 1;

    for(int n = 0; n <= nsteps; n++) {
        if (n % saida == 0 || n == nsteps) {
            double um, vm, mu, mv, e;
            diagnosticos(u, v, N, dx, &um, &vm, &mu, &mv, &e);
            printf("%8.4f %12.6e %12.6e %14.8e %14.8e %14.8e\n",
                   n * dt, um, vm, mu, mv, e);
        }

        if (n == nsteps) break;
        passo(u, v, un, vn, N, r_real);
        double *t;
        t = u; u = un; un = t;
        t = v; v = vn; vn = t;
    }

    /**
     * Validação do modo 2 contra solução analítica:
     * gaussiana 2D difundindo: sigma^2(t) = s0^2 + 2 nu t,
     * u(x, y, t) = A * (s0^2/sigma^2) * exp(-r^2 / (2 sigma^2))
     */
    if (modo == 2) {
        double s2 = s0 * s0 + 2 * nu * tf;
        double err_max = 0, err_l2 = 0, ref_l2 = 0;
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
        double um, vm, mu, mv, e;
        diagnosticos(u, v, N, dx, &um, &vm, &mu, &mv, &e);
        double dev = 0;
        for (int k = 0; k < N * N; k++) {
            double eu = (modo == 1) ? U0 : 0, ev = (modo == 1) ? V0 : 0;
            dev = fmax(dev, fmax(fabs(u[k] - eu), fabs(v[k] - ev)));
        }
        printf("\nDesvio maximo em relacao ao estado inicial: %.3e %s\n", dev,
               dev < 1e-12 ? "(OK: campo estavel)" : "(ATENCAO)");
    }
 
    free(u); free(v); free(un); free(vn);
    return 0;
}