/**TAREFA 17
 * Multiplicacao matriz-vetor com distribuicao por COLUNAS (tipos derivados MPI)
 *
 * compilacao: mpicc -O3 -o tarefa17 tarefa17.c
 * execucao:   mpirun -np P ./tarefa17 M N [REPS]
 * saida CSV:  P,M,N,tempo_total,tempo_calculo
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

/**
 * INICIALIZACAO
 */
static void inicializa_matriz(double *A, int M, int N, int ld)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[(size_t)i * ld + j] = (double)((i + j) % 10 + 1);
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

    /* MPI_Scatter exige blocos iguais -- padding de colunas (zeros) se necessario */
    int colunas_local = (N + P - 1) / P;
    int N_pad = colunas_local * P;

    double *A = NULL, *y = NULL, *x = NULL;
    double *x_local = malloc((size_t)colunas_local * sizeof(double));
    double *A_local = malloc((size_t)M * colunas_local * sizeof(double));
    double *y_parcial = malloc((size_t)M * sizeof(double));

    if (!x_local || !A_local || !y_parcial) {
        fprintf(stderr,
                "Processo %d: falha de alocacao (M=%d N=%d P=%d colunas_local=%d, "
                "A_local=%.1f MB)\n",
                rank, M, N, P, colunas_local,
                (double)M * colunas_local * sizeof(double) / 1e6);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /**
     * TIPO DERIVADO:
     *  - vector: M blocos de 'colunas_local' doubles, com passo N_pad (uma linha inteira)
     *  - resized: extent = colunas_local doubles, para que o i-esimo bloco enviado
     *    pelo Scatter comece na coluna i*colunas_local
     */
    MPI_Datatype tipo_col, tipo_col_resized;
    MPI_Type_vector(M, colunas_local, N_pad, MPI_DOUBLE, &tipo_col);
    MPI_Type_create_resized(tipo_col, 0,
                            (MPI_Aint)(colunas_local * sizeof(double)),
                            &tipo_col_resized);
    MPI_Type_commit(&tipo_col_resized);

    if (rank == 0) {
        A = calloc((size_t)M * N_pad, sizeof(double));
        x = calloc((size_t)N_pad, sizeof(double));
        y = malloc((size_t)M * sizeof(double));
        if (!A || !x || !y) {
            fprintf(stderr, "Processo 0: falha de alocacao de A/x/y.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        inicializa_matriz(A, M, N, N_pad);
        inicializa_vetor(x, N);
    }

    double soma_total = 0.0, soma_calc = 0.0;

    for (int r = 0; r < reps; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        /* Distribui blocos de colunas (1 tipo derivado por processo) */
        MPI_Scatter(A, 1, tipo_col_resized,
                    A_local, M * colunas_local, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        /* Distribui o segmento correspondente de x */
        MPI_Scatter(x, colunas_local, MPI_DOUBLE,
                    x_local, colunas_local, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        /* Calculo: contribuicao parcial para TODOS os elementos de y */
        double tc0 = MPI_Wtime();
        for (int i = 0; i < M; i++) {
            double soma = 0.0;
            for (int j = 0; j < colunas_local; j++)
                soma += A_local[(size_t)i * colunas_local + j] * x_local[j];
            y_parcial[i] = soma;
        }
        double tc1 = MPI_Wtime();

        /* Soma dos vetores parciais no processo 0 */
        MPI_Reduce(y_parcial, y, M, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

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
        /* verificacao contra calculo serial */
        double erro = 0.0;
        for (int i = 0; i < M; i++) {
            double ref = 0.0;
            for (int j = 0; j < N; j++)
                ref += A[(size_t)i * N_pad + j] * x[j];
            double d = fabs(ref - y[i]);
            if (d > erro) erro = d;
        }
        if (erro > 1e-6)
            fprintf(stderr, "ERRO: resultado incorreto (erro max = %g)\n", erro);

        printf("%d,%d,%d,%.6f,%.6f\n", P, M, N, soma_total / reps, soma_calc / reps);

        free(A);
        free(x);
        free(y);
    }

    MPI_Type_free(&tipo_col_resized);
    MPI_Type_free(&tipo_col);
    free(x_local);
    free(A_local);
    free(y_parcial);

    MPI_Finalize();
    return 0;
}