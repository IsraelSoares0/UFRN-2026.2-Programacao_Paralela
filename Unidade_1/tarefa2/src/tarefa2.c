/*
 * mxv.c - Multiplicação Matriz x Vetor (MxV): comparação de padrões de acesso
 *
 * Versão 1 (por linhas):  laço externo percorre linhas i,
 *                          laço interno percorre colunas j
 *                          -> acessa A[i][j] sequencialmente (stride 1)
 *
 * Versão 2 (por colunas):  laço externo percorre colunas j,
 *                          laço interno percorre linhas i
 *                          -> acessa A[i][j] pulando N elementos a cada
 *                             passo (stride N), pois em C a matriz é
 *                             armazenada em ordem row-major.
 *
 * Compilar: gcc -O2 -o mxv mxv.c
 * Executar: ./mxv
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Aloca matriz N x N como um único bloco contíguo (row-major) */
static double **aloca_matriz(int n) {
    double **A = malloc(n * sizeof(double *));
    double *dados = malloc((size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        A[i] = &dados[(size_t)i * n];
    return A;
}

static void libera_matriz(double **A) {
    free(A[0]);
    free(A);
}

static void preenche(double **A, double *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (double)(rand() % 100) / 7.0;
        for (int j = 0; j < n; j++)
            A[i][j] = (double)(rand() % 100) / 3.0;
    }
}

/* Versão por LINHAS: laço interno varia a coluna j (acesso sequencial) */
static void mxv_por_linhas(double **A, double *v, double *r, int n) {
    for (int i = 0; i < n; i++) {
        double soma = 0.0;
        for (int j = 0; j < n; j++)
            soma += A[i][j] * v[j];
        r[i] = soma;
    }
}

/* Versão por COLUNAS: laço interno varia a linha i (acesso com stride n) */
static void mxv_por_colunas(double **A, double *v, double *r, int n) {
    for (int i = 0; i < n; i++)
        r[i] = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            r[i] += A[i][j] * v[j];
}

/* Retorna tempo em segundos, alta resolução */
static double agora(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int tamanhos_padrao[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int n_tam = sizeof(tamanhos_padrao) / sizeof(tamanhos_padrao[0]);
    int repeticoes = 5;

    printf("%10s %14s %14s %10s\n",
           "N", "linhas (s)", "colunas (s)", "razao");
    printf("--------------------------------------------------------\n");

    for (int t = 0; t < n_tam; t++) {
        int n = tamanhos_padrao[t];
        double **A = aloca_matriz(n);
        double *v = malloc(n * sizeof(double));
        double *r = malloc(n * sizeof(double));

        srand(42);
        preenche(A, v, n);

        double melhor_linhas = 1e18, melhor_colunas = 1e18;

        for (int rep = 0; rep < repeticoes; rep++) {
            double t0 = agora();
            mxv_por_linhas(A, v, r, n);
            double t1 = agora();
            if (t1 - t0 < melhor_linhas) melhor_linhas = t1 - t0;
        }

        for (int rep = 0; rep < repeticoes; rep++) {
            double t0 = agora();
            mxv_por_colunas(A, v, r, n);
            double t1 = agora();
            if (t1 - t0 < melhor_colunas) melhor_colunas = t1 - t0;
        }

        printf("%10d %14.6f %14.6f %10.2fx\n",
               n, melhor_linhas, melhor_colunas,
               melhor_colunas / melhor_linhas);

        libera_matriz(A);
        free(v);
        free(r);
    }

    return 0;
}
