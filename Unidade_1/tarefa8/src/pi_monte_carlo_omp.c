/* =====================================================================
 * Estimativa estocastica de PI (metodo de Monte Carlo) com OpenMP
 *
 * Sao implementadas 4 versoes:
 *   1) rand()      + acumulacao via variavel privada + #pragma omp critical
 *   2) rand()      + acumulacao via vetor compartilhado + soma serial
 *   3) xorshift32() + acumulacao via variavel privada + #pragma omp critical
 *   4) xorshift32() + acumulacao via vetor compartilhado + soma serial
 *
 * Como o ambiente e Windows (sem rand_r() do POSIX), o gerador rand_r()
 * foi substituido por um Xorshift32 thread-safe (estado privado por
 * thread), que cumpre o mesmo papel: gerar numeros aleatorios sem
 * depender de um estado global compartilhado.
 *
 * Compilar (MinGW / GCC no Windows):
 *   gcc -O2 -fopenmp pi_monte_carlo_omp.c -o pi_monte_carlo_omp.exe
 *
 * Executar:
 *   pi_monte_carlo_omp.exe [num_pontos] [num_threads]
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

/* ---------------------------------------------------------------------
 * VERSAO 1: rand() + contador privado + acumulação global via critical
 * --------------------------------------------------------------------- */
double pi_rand_critical(long long n_pontos, int n_threads)
{
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

/* ---------------------------------------------------------------------
 * VERSAO 2: rand() + vetor compartilhado + soma serial apos a regial paralela
 * --------------------------------------------------------------------- */
double pi_rand_vetor(long long n_pontos, int n_threads)
{
    long long acertos_por_thread[MAX_THREADS] = {0};

    #pragma omp parallel num_threads(n_threads)
    {
        int id = omp_get_thread_num();
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

        acertos_por_thread[id] = acertos_local;
    }

    long long total_acertos = 0;
    for (int i = 0; i < n_threads; i++)
    {
        total_acertos += acertos_por_thread[i];
    }

    return 4.0 * (double)total_acertos / (double)n_pontos;
}

/* ---------------------------------------------------------------------
 * VERSAO 3: xorshift32() + contador privado + acumulador global via critical
 * --------------------------------------------------------------------- */
double pi_xorshift_critical(long long n_pontos, int n_threads)
{
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

/* ---------------------------------------------------------------------
 * VERSAO 4: xorshift32() + vetor compartilhado + soma serial
 * --------------------------------------------------------------------- */
double pi_xorshift_vetor(long long n_pontos, int n_threads)
{
    long long acertos_por_thread[MAX_THREADS] = {0};

    #pragma omp parallel num_threads(n_threads)
    {
        int id = omp_get_thread_num();
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

        acertos_por_thread[id] = acertos_local;
    }

    long long total_acertos = 0;
    for (int i = 0; i < n_threads; i++)
    {
        total_acertos += acertos_por_thread[i];
    }

    return 4.0 * (double)total_acertos / (double)n_pontos;
}

/* ---------------------------------------------------------------------
 * MAIN: executa e cronometra as 4 versoes
 * --------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    long long n_inicial = 100000;
    long long n_final = 1000000000;
    int n_threads = omp_get_max_threads();

    if (argc >= 2) n_inicial = atoll(argv[1]);
    if (argc >= 3) n_final = atoll(argv[2]);
    if (argc >= 4) n_threads = atoi(argv[3]);

    if (n_inicial <= 0 || n_final < n_inicial)
    {
        fprintf(stderr, "Intervalo de pontos invalido.\n");
        fprintf(stderr, "Uso: %s [n_inicial] [n_final] [n_threads]\n", argv[0]);
        return 1;
    }

    if (n_threads <= 0)
    {
        fprintf(stderr, "Numero de threads invalido.\n");
        return 1;
    }

    if (n_threads > MAX_THREADS)
    {
        fprintf(stderr, "Numero de threads limitado a %d.\n", MAX_THREADS);
        n_threads = MAX_THREADS;
    }

    srand((unsigned int)time(NULL));

    printf("===============================================================\n");
    printf("        ESTIMATIVA DE PI - METODO DE MONTE CARLO\n");
    printf("===============================================================\n");
    printf("PI_REAL = %.15f\n", PI_REAL);
    printf("Threads = %d\n", n_threads);
    printf("Intervalo de n_pontos = %lld ate %lld\n\n", n_inicial, n_final);

    printf("%-12s %-8s %-12s %-12s %-12s %-12s\n",
           "N_PONTOS", "VERSAO", "PI_MEDIDO", "ERRO_ABS", "ERRO_REL(%)", "TEMPO(s)");
    printf("--------------------------------------------------------------------------\n");

    for (long long n_pontos = n_inicial;
         n_pontos <= n_final;
         )
    {
        double pi_medido;
        double erro_abs;
        double erro_rel;
        double t0, t1;

        /* =========================================================
         * VERSAO 1: rand() + critical
         * ========================================================= */
        t0 = omp_get_wtime();
        pi_medido = pi_rand_critical(n_pontos, n_threads);
        t1 = omp_get_wtime();

        erro_abs = fabs(pi_medido - PI_REAL);
        erro_rel = (erro_abs / PI_REAL) * 100.0;

        printf("%-12lld %-8d %-12.8f %-12.8f %-12.6f %-12.4f\n",
               n_pontos, 1, pi_medido, erro_abs, erro_rel, t1 - t0);

        /* =========================================================
         * VERSAO 2: rand() + vetor
         * ========================================================= */
        t0 = omp_get_wtime();
        pi_medido = pi_rand_vetor(n_pontos, n_threads);
        t1 = omp_get_wtime();

        erro_abs = fabs(pi_medido - PI_REAL);
        erro_rel = (erro_abs / PI_REAL) * 100.0;

        printf("%-12lld %-8d %-12.8f %-12.8f %-12.6f %-12.4f\n",
               n_pontos, 2, pi_medido, erro_abs, erro_rel, t1 - t0);

        /* =========================================================
         * VERSAO 3: xorshift32() + critical
         * ========================================================= */
        t0 = omp_get_wtime();
        pi_medido = pi_xorshift_critical(n_pontos, n_threads);
        t1 = omp_get_wtime();

        erro_abs = fabs(pi_medido - PI_REAL);
        erro_rel = (erro_abs / PI_REAL) * 100.0;

        printf("%-12lld %-8d %-12.8f %-12.8f %-12.6f %-12.4f\n",
               n_pontos, 3, pi_medido, erro_abs, erro_rel, t1 - t0);

        /* =========================================================
         * VERSAO 4: xorshift32() + vetor
         * ========================================================= */
        t0 = omp_get_wtime();
        pi_medido = pi_xorshift_vetor(n_pontos, n_threads);
        t1 = omp_get_wtime();

        erro_abs = fabs(pi_medido - PI_REAL);
        erro_rel = (erro_abs / PI_REAL) * 100.0;

        printf("%-12lld %-8d %-12.8f %-12.8f %-12.6f %-12.4f\n",
               n_pontos, 4, pi_medido, erro_abs, erro_rel, t1 - t0);

        printf("--------------------------------------------------------------------------\n");

        /* Evita overflow ao multiplicar por 10. */
        if (n_pontos > n_final / 10)
            break;

        n_pontos *= 10;
    }

    printf("\nLegenda:\n");
    printf("  ERRO_ABS    = |PI_MEDIDO - PI_REAL|\n");
    printf("  ERRO_REL    = (ERRO_ABS / PI_REAL) * 100\n");

    return 0;
}
