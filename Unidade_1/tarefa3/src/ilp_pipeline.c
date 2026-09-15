/*
 * Compile com diferentes níveis de otimização e compare os tempos:
 *
 *   gcc -O0 -o ilp_O0 ilp_pipeline.c
 *   gcc -O2 -o ilp_O2 ilp_pipeline.c
 *   gcc -O3 -o ilp_O3 ilp_pipeline.c
 *
 * Execute várias vezes cada binário para obter uma média confiável:
 *
 *   ./ilp_O0
 *   ./ilp_O2
 *   ./ilp_O3
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000   /* tamanho do vetor (100 milhoes de elementos) */
#define REPS 5

double medir_tempo(struct timespec inicio, struct timespec fim) {
    return (fim.tv_sec - inicio.tv_sec) +
           (fim.tv_nsec - inicio.tv_nsec) / 1e9;
}

// Laço 1: inicialização do vetor

void inicializa_vetor(double *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (double)i * 0.5 + 1.0;
    }
}

// Laço 2: soma acumulativa com dependência (uma única variável)
 
double soma_dependente(double *v, int n) {
    double soma = 0.0;
    for (int i = 0; i < n; i++) {
        soma += v[i];   /* soma[i] depende de soma[i-1] */
    }
    return soma;
}

// Laço 3: soma com múltiplos acumuladores (quebra a dependência)

double soma_independente(double *v, int n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i;

    /* processa de 4 em 4 elementos */
    for (i = 0; i + 3 < n; i += 4) {
        s0 += v[i];
        s1 += v[i + 1];
        s2 += v[i + 2];
        s3 += v[i + 3];
    }

    /* trata elementos restantes, caso n não seja múltiplo de 4 */
    for (; i < n; i++) {
        s0 += v[i];
    }

    return (s0 + s1) + (s2 + s3);
}

int main(void) {
    double *v = malloc(N * sizeof(double));
    if (!v) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        return 1;
    }

    struct timespec t0, t1;
    double tempo_init = 0, tempo_dep = 0, tempo_indep = 0;
    double resultado_dep = 0, resultado_indep = 0;

    printf("Vetor de %d elementos, %d repeticoes por laço\n\n", N, REPS);

    /* ---- Laço 1: inicialização ---- */
    for (int r = 0; r < REPS; r++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        inicializa_vetor(v, N);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        tempo_init += medir_tempo(t0, t1);
    }
    tempo_init /= REPS;

    /* ---- Laço 2: soma dependente ---- */
    for (int r = 0; r < REPS; r++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        resultado_dep = soma_dependente(v, N);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        tempo_dep += medir_tempo(t0, t1);
    }
    tempo_dep /= REPS;

    /* ---- Laço 3: soma independente (múltiplos acumuladores) ---- */
    for (int r = 0; r < REPS; r++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        resultado_indep = soma_independente(v, N);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        tempo_indep += medir_tempo(t0, t1);
    }
    tempo_indep /= REPS;

    printf("Laco 1 (inicializacao)........: %.4f s\n", tempo_init);
    printf("Laco 2 (soma dependente)......: %.4f s   (resultado = %.2f)\n",
           tempo_dep, resultado_dep);
    printf("Laco 3 (soma c/ 4 acumuladores): %.4f s   (resultado = %.2f)\n",
           tempo_indep, resultado_indep);

    printf("\nSpeedup do laco 3 em relacao ao laco 2: %.2fx\n",
           tempo_dep / tempo_indep);

    free(v);
    return 0;
}