/* =====================================================================
 * Tarefa 10: Mecanismos de Sincronizacao com OpenMP
 * Estimativa estocastica de PI (metodo de Monte Carlo)
 *
 * Este arquivo reimplementa o estimador da Tarefa 8
 * (pi_monte_carlo_omp.c) para comparar 5 estrategias de sincronizacao
 * entre threads na hora de somar os "acertos" (pontos dentro do
 * circulo):
 *
 *   V1) Contador PRIVADO + rand()      + soma final via #pragma omp critical
 *       (equivalente a versao 1 da Tarefa 8 - referencia/baseline)
 *
 *   V2) Contador PRIVADO + xorshift32  + soma final via #pragma omp critical
 *       (equivalente a versao 3 da Tarefa 8 - referencia/baseline)
 *
 *   V3) Contador COMPARTILHADO + xorshift32 + incremento protegido por
 *       #pragma omp critical a CADA acerto                (NOVA)
 *
 *   V4) Contador COMPARTILHADO + xorshift32 + incremento protegido por
 *       #pragma omp atomic a CADA acerto                  (NOVA)
 *
 *   V5) Acumulacao via clausula reduction(+:...) + xorshift32,
 *       sem nenhuma diretiva explicita de exclusao mutua   (NOVA)
 *
 * Compilar (gcc/MinGW-w64):
 *   gcc -O2 -fopenmp pi_monte_carlo_sync.c -o pi_monte_carlo_sync.exe
 *
 * Executar:
 *   pi_monte_carlo_sync.exe [n_inicial] [n_final] [n_threads]
 * ===================================================================== */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <omp.h>

#define PI_REAL 3.14159265358979323846

#define MAX_THREADS 128

/* --------------------------------------------------
 * Gerador Xorshift32 (thread-safe)
 * -------------------------------------------------- */
static inline unsigned int xorshift32(unsigned int *state)
{
    unsigned int x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;
    return x;
}

/* Geracao de double aleatorio */
static inline double random_double(unsigned int *state)
{
    return (double)xorshift32(state) / (double)UINT_MAX;
}

/* Cria uma semente diferente por thread e NUNCA igual a zero */
static inline unsigned int semente_da_thread(void)
{
    unsigned int base = (unsigned int)time(NULL) ^
                         ((unsigned int)omp_get_thread_num() * 2654435761u);
    return base | 1u;
}

/* ================================================================
 * V1: Contador PRIVADO + rand() + soma final via critical
 * (Versão 1 da tarefa 8)
 * ================================================================ */
double pi_privado_critical_rand(long long n_pontos, int n_threads) {
    long long total_acertos = 0;

    #pragma omp parallel num_threads(n_threads)
    {
        long long acertos_local = 0;
        long long i;

        #pragma omp for
        for (i = 0; i < n_pontos; i++)
        {
            double x = (double)rand() / (double)RAND_MAX;
            double y = (double)rand() / (double)RAND_MAX;

            if (x * x + y * y <= 1.0)
            {
                acertos_local++;
            }
        }

        /* Acumulação do total */
        #pragma omp critical
        {
            total_acertos += acertos_local;
        }
    }

    return 4.0 * (double)total_acertos / (double)n_pontos;
}

/* ================================================================
 * V2: Contador PRIVADO + xorshift32 + soma final via critical
 * (Versão 3 da tarefa 8)
 * ================================================================ */
double pi_privado_critical_xorshift(long long n_pontos, int n_threads) {
    long long total_acertos = 0;

    #pragma omp parallel num_threads(n_threads)
    {
        unsigned int seed = semente_da_thread();
        long long acertos_local = 0;
        long long i;

        #pragma omp for
        for (i = 0; i < n_pontos; i++)
        {
            double x = random_double(&seed);
            double y = random_double(&seed);

            if (x * x + y * y <= 1.0)
            {
                acertos_local++;
            }
        }

        #pragma omp critical
        {
            total_acertos += acertos_local;
        }
    }

    return 4.0 * (double)total_acertos / (double)n_pontos;
}

/* ================================================================
 * V3: Contador COMPARTILHADO + xorshift + critical a cada acerto
 * ================================================================ */
double pi_compartilhado_critical_xorshift32(long long n_pontos, int n_threads) {
    long long total_acertos = 0;

    #pragma omp parallel num_threads(n_threads)
    {
        unsigned int seed = semente_da_thread();
        long long i;

        #pragma omp for
        for (i = 0; i < n_pontos; i++) {
            double x = random_double(&seed);
            double y = random_double(&seed);

            if (x * x + y * y <= 1.0)
            {
                #pragma omp critical
                {
                    total_acertos++;
                }
            }
        }
    }

    return 4.0 * (double)total_acertos / (double)n_pontos;
}

/* ================================================================
 * V4: Contador COMPARTILHADO + xorshift + atomic a cada acerto
 * ================================================================ */
double pi_compartilhado_atomic_xorshift32(long long n_pontos, int n_threads) {
    long long total_acertos = 0;

    #pragma omp parallel num_threads(n_threads)
    {
        unsigned int seed = semente_da_thread();
        long long i;

        #pragma omp for
        for (i = 0; i < n_pontos; i++) {
            double x = random_double(&seed);
            double y = random_double(&seed);

            if (x * x + y * y <= 1.0)
            {
                #pragma omp atomic
                total_acertos++;
            }
        }
    }

    return 4.0 * (double)total_acertos / (double)n_pontos;
}

/* ================================================================
 * V5: xorshift32 + clausula reduction(+:...)
 * ================================================================ */
double pi_reduction_xorshift32(long long n_pontos, int n_threads) {
    long long total_acertos = 0;

    #pragma omp parallel num_threads(n_threads)
    {
        unsigned int seed = semente_da_thread();
        long long i;

        #pragma omp for reduction(+:total_acertos)
        for (i = 0; i < n_pontos; i++) {
            double x = random_double(&seed);
            double y = random_double(&seed);

            if (x * x + y * y <= 1.0)
            {
                total_acertos++;
            }
        }
    }

    return 4.0 * (double)total_acertos / (double)n_pontos;
}

/* ================================================================
 * Estrutura auxiliar para rodar/computar erro/imprimir cada versao
 * ================================================================ */
typedef double (*pi_func_t)(long long, int);

typedef struct {
    const char *nome;
    pi_func_t func;
} versao_t;

static void executar_versao(const char *nome, pi_func_t f, long long n_pontos, int n_threads) {
    double t0 = omp_get_wtime();
    double pi_medido = f(n_pontos, n_threads);
    double t1 = omp_get_wtime();

    double erro_abs = fabs(pi_medido - PI_REAL);
    double erro_rel = (erro_abs / PI_REAL) * 100.0;

    printf("%-12lld %-34s %-12.8f %-12.8f %-12.6f %-12.4f\n",
           n_pontos, nome, pi_medido, erro_abs, erro_rel, t1 - t0);
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main(int argc, char *argv[]) {
    long long n_inicial = 10000;
    long long n_final = 1000000000;
    int n_threads = omp_get_max_threads();

    if (argc >= 2) n_inicial = atoll(argv[1]);
    if (argc >= 3) n_final = atoll(argv[2]);
    if (argc >= 4) n_threads = atoi(argv[3]);

    if (n_inicial <= 0 || n_final < n_inicial) {
        fprintf(stderr, "Intervalo de pontos invalido.\n");
        fprintf(stderr, "Uso: %s [n_inicial] [n_final] [n_threads]\n", argv[0]);
        return 1;
    }

    if (n_threads <= 0) {
        fprintf(stderr, "Numero de threads invalido.\n");
        return 1;
    }

    if (n_threads > MAX_THREADS) {
        fprintf(stderr, "Numero de threads limitado a %d.\n", MAX_THREADS);
        n_threads = MAX_THREADS;
    }

    srand((unsigned int)time(NULL));

    printf("===============================================================\n");
    printf("   TAREFA 10 - MECANISMOS DE SINCRONIZACAO COM OpenMP (PI)\n");
    printf("===============================================================\n");
    printf("PI_REAL = %.15f\n", PI_REAL);
    printf("Threads = %d\n", n_threads);
    printf("Intervalo de n_pontos = %lld ate %lld\n\n", n_inicial, n_final);
 
    printf("%-12s %-34s %-12s %-12s %-12s %-12s\n",
           "N_PONTOS", "VERSAO", "PI_MEDIDO", "ERRO_ABS", "ERRO_REL(%)", "TEMPO(s)");
    printf("------------------------------------------------\n");
 
    versao_t versoes[] = {
        { "V1 privado+rand()+critical",          pi_privado_critical_rand             },
        { "V2 privado+xorshift+critical",        pi_privado_critical_xorshift         },
        { "V3 compartilhado+xorshift+critical",  pi_compartilhado_critical_xorshift32 },
        { "V4 compartilhado+xorshift+atomic",    pi_compartilhado_atomic_xorshift32   },
        { "V5 xorshift+reduction",               pi_reduction_xorshift32              },
    };
    int n_versoes = sizeof(versoes) / sizeof(versoes[0]);
 
    for (long long n_pontos = n_inicial; n_pontos <= n_final; )
    {
        for (int v = 0; v < n_versoes; v++)
            executar_versao(versoes[v].nome, versoes[v].func, n_pontos, n_threads);
 
        printf("--------------------------------------------\n");
 
        if (n_pontos > n_final / 10) break;
        n_pontos *= 10;
    }
 
    printf("\nLegenda:\n");
    printf("  ERRO_ABS = |PI_MEDIDO - PI_REAL|\n");
    printf("  ERRO_REL = (ERRO_ABS / PI_REAL) * 100\n");
 
    return 0;
}