/**TAREFA 16
 * 
 * compilacao: mpicc -O3 -o tarefa16 tarefa16.c
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

/**
 * INICIALIZACAO
 */
static void inicializa_matriz(double *A, int M, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[(size_t)i * N + j] = (double)((i + j) % 10 + 1);
}

static void inicializa_vetor(double *x, int N)
{
    for (int i = 0; i < N; i++)
        x[i] = (double)(i % 5 + 1);
}

/**
 * MAIN
 */
int main(int argc, char **argv)
{
    int rank, P;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <M> <N> [REPS]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int reps = (argc > 3) ? atoi(argv[3]) : 5;

    if (M <= 0 || N <= 0 || reps <= 0) {
        if (rank == 0)
            fprintf(stderr, "Parametros invalidos.\n");
        MPI_Finalize();
        return 1;
    }

    /* MPI_Scatter exige blocos iguais -- Padding se necessario */
    int linhas_local = (M + P - 1) / P;
    int M_pad = linhas_local * P;

    double *A = NULL, *y = NULL;
    double *x = malloc((size_t)N * sizeof(double));
    double *A_local = malloc((size_t)linhas_local * N * sizeof(double));
    double *y_local = malloc((size_t)linhas_local * sizeof(double));

    if (!x || !A_local || !y_local) {
        fprintf(stderr,
                "Processo %d: falha de alocacao (x=%p A_local=%p y_local=%p | "
                "M=%d N=%d P=%d linhas_local=%d, A_local=%.1f MB)\n",
                rank, (void *)x, (void *)A_local, (void *)y_local,
                M, N, P, linhas_local,
                (double)linhas_local * N * sizeof(double) / 1e6);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = calloc((size_t)M_pad * N, sizeof(double));
        y = malloc((size_t)M_pad * sizeof(double));
        if (!A || !y) {
            fprintf(stderr, "Processo 0: falha de alocacao de A/y.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        inicializa_matriz(A, M, N);
        inicializa_vetor(x, N);
    }

    double soma_total = 0.0, soma_calc = 0.0;

    for (int r = 0; r < reps; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        /* Distribuiu linhas */
        MPI_Scatter(A, linhas_local * N, MPI_DOUBLE,
                    A_local, linhas_local * N, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        /* Distribui x inteiro */
        MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Calculo */
        double tc0 = MPI_Wtime();
        for (int i = 0; i < linhas_local; i++) {
            double soma = 0.0;
            for (int j = 0; j < N; j++)
                soma += A_local[(size_t)i * N + j] * x[j];
            y_local[i] = soma;
        }
        double tc1 = MPI_Wtime();

        /* Coleta no processo 0 */
        MPI_Gather(y_local, linhas_local, MPI_DOUBLE,
                   y, linhas_local, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        double t1 = MPI_Wtime();

        /* tempo do experimento */

        double t_total_loc = t1 - t0, t_calc_loc = tc1 - tc0;
        double t_total, t_calc;

        MPI_Reduce(&t_total_loc, &t_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_calc_loc, &t_calc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            soma_total += t_total;
            soma_calc += t_calc;
        }
    }

    if (rank == 0) {
        printf("%d,%d,%d,%.6f,%.6f\n", P, M, N, soma_total / reps, soma_calc / reps);

        free(A);
        free(y);
    }
    
    free(x);
    free(A_local);
    free(y_local);

    MPI_Finalize();
    return 0;
}