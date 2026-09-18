/*
 * Simulação de difusão de calor em uma barra 1D DOBRADA EM ANEL
 * Método: Diferenças Finitas Explícito (FTCS - Forward Time, Centered Space)
 * Condição de contorno: PERIÓDICA (o último ponto é vizinho do primeiro)
 *
 * Equação resolvida:  du/dt = alpha * d²u/dx²
 *
 * Condição inicial: perfil em cosseno, mais quente no centro da barra
 * e mais frio nas extremidades — e como as extremidades se encontram
 * (formando o anel), a temperatura já "fecha" nesse ponto.
 *
 * Compilar: gcc barra_1d.c -o barra_1d -lm
 * Executar: ./difusao
 *
 * Saída: arquivo "temperaturas.csv" com a temperatura ao longo do anel
 *        em diferentes instantes de tempo (uma linha por instante salvo).
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Parâmetros de simulação */
#define N_PONTOS 100 /* Número de pontos na barra */
#define COMPRIMENTO 1.0 /* Comprimento da barra (m) */
#define ALPHA 0.01 /* Difusividade térmica (m²/s) */
#define T_TOTAL 2.0 /* Tempo total de simulação (s) */
#define TEMP_BASE 20.0 /* temperatura nas bordas/base (ºC) */
#define TEMP_CENTRO 100.0 /* temperatura no centro da barra (ºC) */
#define N_SALVAS 10 /* quantos instantes de tempo salvar no CSV */

int main(void) {
    double dx = COMPRIMENTO / N_PONTOS;

    /* Criterio de estabilidade (numero de Fourier <= 0.5 )*/
    double dt_max = 0.5 * dx * dx / ALPHA;
    double dt = 0.9 * dt_max;

    int n_passos = (int) ceil(T_TOTAL / dt);
    dt = T_TOTAL / n_passos;

    double r = ALPHA * dt / (dx * dx);

    printf("Parametros da simulacao:\n");
    printf("  Pontos na barra : %d\n", N_PONTOS);
    printf("  dx              : %.6f m\n", dx);
    printf("  dt              : %.6e s\n", dt);
    printf("  Passos de tempo : %d\n", n_passos);
    printf("  Numero de Fourier (r = alpha*dt/dx^2) = %.4f (<= 0.5)\n\n", r);

    double *u_atual = malloc(N_PONTOS * sizeof(double));
    double *u_novo = malloc(N_PONTOS * sizeof(double));
    if (!u_atual || !u_novo) {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        return 1;
    }

    for (int i = 0; i < N_PONTOS; i++) {
        double x = i * dx;
        double fase = 2.0 * M_PI * (x - COMPRIMENTO / 2.0) / COMPRIMENTO;
        u_atual[i] = TEMP_BASE + (TEMP_CENTRO - TEMP_BASE) * (cos(fase) + 1.0) / 2.0;
    }

    FILE *arquivo = fopen("temperaturas.csv", "w");
    if (!arquivo) {
        fprintf(stderr, "Erro ao criar arquivo de saida.\n");
        free(u_atual);
        free(u_novo);
        return 1;
    }

    fprintf(arquivo, "tempo");
    for (int i = 0; i < N_PONTOS; i++) {
        fprintf(arquivo, ",%.6f", i * dx);
    }
    fprintf(arquivo, "\n");

    int intervalo_salvar = n_passos / N_SALVAS;
    if (intervalo_salvar < 1) intervalo_salvar = 1;

    fprintf(arquivo, "%.6f", 0.0);
    for (int i = 0; i < N_PONTOS; i++) {
        fprintf(arquivo, ",%.6f", u_atual[i]);
    }
    fprintf(arquivo, "\n");

    for (int passo = 1; passo <= n_passos; passo++) {
        for (int i = 0; i < N_PONTOS; i++) {
            int direita = (i + 1) % N_PONTOS;
            int esquerda = (i - 1 + N_PONTOS) % N_PONTOS;
            u_novo[i] = u_atual[i] + r * (u_atual[direita] - 2.0 * u_atual[i] + u_atual[esquerda]);
        }

        double *tmp = u_atual;
        u_atual = u_novo;
        u_novo = tmp;

        if (passo % intervalo_salvar == 0 || passo == n_passos) {
            double t = passo * dt;
            fprintf(arquivo, "%.6f", t);
            for (int i = 0; i < N_PONTOS; i++) {
                fprintf(arquivo, ",%.6f", u_atual[i]);
            }
            fprintf(arquivo, "\n");
        }
    }

    fclose(arquivo);
    free(u_atual);
    free(u_novo);

    printf("Simulacao concluida. Resultados salvos em 'temperaturas.csv'.\n");
    return 0;
}