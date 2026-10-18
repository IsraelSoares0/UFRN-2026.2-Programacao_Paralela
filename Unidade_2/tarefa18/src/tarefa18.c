/*
 * Tarefa 18 - Adição de vetores: CPU (OpenMP) x GPU (OpenMP target)
 *
 * Compilação (exemplos):
 *   GCC (nvptx)    : gcc -fopenmp -foffload=nvptx-none -O3 tarefa18.c -o tarefa18
 *   Sem GPU        : gcc -fopenmp -O3 tarefa18.c -o tarefa18
 *
 * Uso: ./tarefa18 [N_max] [repeticoes]
 * Saída: resultados.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define REL_TOL 1e-6f

/* Soma de vetores somente na CPU */
void soma_cpu(const float *a, const float *b, float *c, long n)
{
    #pragma omp parallel for
    for (long i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* Soma de vetores na GPU */
void soma_gpu(const float *a, const float *b, float *c, long n)
{
    #pragma omp target parallel for map(to: a[0:n], b[0:n]) map(from: c[0:n])
    for (long i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* Conta elementos com erro relativo acima da tolerância (esperado: 6*i) */
long verifica(const float *c, long n)
{
    long erros = 0;
    #pragma omp parallel for reduction(+:erros)
    for (long i = 0; i < n; i++) {
        float esperado = 6.0f * (float)i;
        if (fabsf(c[i] - esperado) > REL_TOL * fabsf(esperado)) erros++;
    }
    return erros;
}

int main(int argc, char **argv)
{
    long n_max = (argc > 1) ? atol(argv[1]) : 100000000L;
    int  reps  = (argc > 2) ? atoi(argv[2]) : 5;

    long tamanhos[] = {10000L, 100000L, 1000000L, 5000000L, 10000000L,
                       50000000L, 100000000L, 200000000L};
    int  ntam = sizeof(tamanhos) / sizeof(tamanhos[0]);

    printf("Threads CPU: %d | Dispositivos GPU: %d\n",
           omp_get_max_threads(), omp_get_num_devices());

    FILE *csv = fopen("resultados.csv", "w");
    if (!csv) { perror("resultados.csv"); return 1; }
    fprintf(csv, "N,repeticao,tempo_cpu_s,tempo_gpu_s,speedup,erros_cpu,erros_gpu\n");

    for (int t = 0; t < ntam; t++) {
        long n = tamanhos[t];
        if (n > n_max) break;

        float *a = malloc(n * sizeof(float));
        float *b = malloc(n * sizeof(float));
        float *c = malloc(n * sizeof(float));
        if (!a || !b || !c) {
            fprintf(stderr, "Sem memória para N=%ld\n", n);
            free(a); free(b); free(c);
            break;
        }

        #pragma omp parallel for
        for (long i = 0; i < n; i++) {
            a[i] = (float)i;
            b[i] = 5.0f * (float)i;
        }

        for (int r = 1; r <= reps; r++) {
            double t0, t_cpu, t_gpu;

            t0 = omp_get_wtime();
            soma_cpu(a, b, c, n);
            t_cpu = omp_get_wtime() - t0;
            long err_cpu = verifica(c, n);

            #pragma omp parallel for
            for (long i = 0; i < n; i++) c[i] = 0.0f;

            t0 = omp_get_wtime();
            soma_gpu(a, b, c, n);
            t_gpu = omp_get_wtime() - t0;
            long err_gpu = verifica(c, n);

            fprintf(csv, "%ld,%d,%.6f,%.6f,%.3f,%ld,%ld\n",
                    n, r, t_cpu, t_gpu, t_cpu / t_gpu, err_cpu, err_gpu);
            printf("N=%-10ld rep=%d  CPU=%.5fs  GPU=%.5fs  erros=%ld/%ld\n",
                   n, r, t_cpu, t_gpu, err_cpu, err_gpu);
        }
        free(a); free(b); free(c);
    }

    fclose(csv);
    printf("Dados salvos em resultados.csv\n");
    return 0;
}