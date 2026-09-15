/*
    Implemente um programa em C que conte quantos números primos existem entre 2 e um valor máximo n.
    Depois, paralelize o laço principal usando a diretiva:
        #pragma omp parallel for
    sem alterar a lógica original. Compare o tempo de execução e os resultados das versões sequênciais
    e paralela. Observe possíveis diferenças no resultado e no desempenho, e reflita sobre os
    desafios iniciais da programação paralela, como correção e distribuição de carga.

    Compilação:
    $ gcc -fopenmp tarefa5.c -o tarefa5
*/

#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

bool verificar_primo(int num) {
    if (num < 2) return false;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    int valores_n[] = {1000000, 5000000, 10000000, 15000000, 20000000};
    int qtd_valores = sizeof(valores_n) / sizeof(valores_n[0]);

    printf("Nucleos disponiveis: %d\n\n", omp_get_num_procs());

    for (int idx = 0; idx < qtd_valores; idx++) {
        int n = valores_n[idx];
        int contagem_seq = 0;
        int contagem_par = 0;

        printf("===================================================\n");
        printf("Intervalo: [2, %d]\n\n", n);

        // ---------- VERSÃO SEQUENCIAL ----------
        double inicio_seq = omp_get_wtime();
        for (int i = 2; i <= n; i++) {
            if (verificar_primo(i)) {
                contagem_seq++;
            }
        }
        double fim_seq = omp_get_wtime();

        // ---------- VERSÃO PARALELA ----------
        double inicio_par = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 2; i <= n; i++) {
            if (verificar_primo(i)) {
                contagem_par++;   // Condição de corrida ?
            }
        }
        double fim_par = omp_get_wtime();

        printf("Sequencial : %d primos encontrados | tempo: %.4f s\n",
               contagem_seq, fim_seq - inicio_seq);

        printf("Paralelo   : %d primos encontrados | tempo: %.4f s\n",
               contagem_par, fim_par - inicio_par);

        if (contagem_seq != contagem_par) {
            printf("\n[ALERTA] Os resultados divergem! Diferenca = %d\n",
                   contagem_seq - contagem_par);
        } else {
            printf("\nOs resultados coincidiram nesta execucao (pode nao ocorrer sempre).\n");
        }

        printf("\n");
    }

    return 0;
}