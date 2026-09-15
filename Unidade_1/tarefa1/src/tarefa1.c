#include <stdio.h>
#include <math.h>
#include <time.h>

double gregory_leibniz(int k) {
    double aux = 0;
    double sinal = 1.0;

    for (int i = 0; i < k; i++) {
        aux += sinal / (2*i + 1);
        sinal = -sinal;
    }

    return 4 * aux;
}

int main(void) {
    const double PI_REAL = 3.14159265358979323846;

    int iteracoes[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    int n = sizeof(iteracoes) / sizeof(iteracoes[0]);

    printf("%-12s %-15s %-15s %-15s\n", "Iteracoes", "Pi calculado", "Erro absoluto", "Tempo (s)");

    for (int idx = 0; idx < n; idx++) {
        int k = iteracoes[idx];

        clock_t inicio = clock();
        double pi = gregory_leibniz(k);
        clock_t fim = clock();

        double tempo = (double)(fim - inicio) / CLOCKS_PER_SEC;
        double erro = fabs(PI_REAL - pi);

        printf("%-12d %-15.10f %-15.10f %-15.6f\n", k, pi, erro, tempo);
    }

    return 0;
}