#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <limits.h>

#define PI_REAL 3.14159265358979323846

/* Gerador Xorshift32 */
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

/* =====================================================================
   Estimativa de pi por Monte Carlo:
   gera pontos aleatorios em [-1,1] x [-1,1] e verifica quantos caem
   dentro do circulo unitario. A razao (pontos_dentro / total) * 4
   aproxima pi (area do circulo / area do quadrado = pi/4).
   ===================================================================== */

/* ---------- Versao sequencial (referencia, sem OpenMP) ---------- */
double pi_sequencial(long n_pontos, unsigned int seed) {
    seed |= 1u; /* garante estado inicial != 0 */
    long dentro = 0;
    for (long i = 0; i < n_pontos; i++) {
        double x = random_double(&seed) * 2.0 - 1.0;
        double y = random_double(&seed) * 2.0 - 1.0;
        if (x * x + y * y <= 1.0) dentro++;
    }
    return 4.0 * (double)dentro / (double)n_pontos;
}

/* ---------- Versao 1: paralela -> CONDICAO DE CORRIDA (intencional) ---------- */
double pi_paralelo_com_erro(long n_pontos) {
    long dentro = 0; /* variavel compartilhada por padrao (default(shared)) */

    #pragma omp parallel for
    for (long i = 0; i < n_pontos; i++) {
        /* rand() usa estado global interno: nao e thread-safe de proposito,
           para preservar o efeito didatico da condicao de corrida */
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0) {
            dentro++;   /* leitura + incremento + escrita: NAO e atomico */
        }
    }
    return 4.0 * (double)dentro / (double)n_pontos;
}

/* ---------- Versao 2: corrigida com #pragma omp critical ---------- */
double pi_paralelo_critical(long n_pontos) {
    long dentro = 0;

    #pragma omp parallel default(none) shared(n_pontos, dentro)
    {
        unsigned int seed = semente_da_thread(); /* uma vez por thread */

        #pragma omp for
        for (long i = 0; i < n_pontos; i++) {
            double x = random_double(&seed) * 2.0 - 1.0;
            double y = random_double(&seed) * 2.0 - 1.0;
            if (x * x + y * y <= 1.0) {
                #pragma omp critical
                dentro++;   /* so uma thread por vez executa o incremento */
            }
        }
    }
    return 4.0 * (double)dentro / (double)n_pontos;
}

/* ---------- Versao 3: reestruturada com omp parallel + omp for ---------- */
double pi_paralelo_otimizado(long n_pontos) {
    long dentro_total = 0;

    #pragma omp parallel default(none) shared(n_pontos, dentro_total)
    {
        long dentro_local = 0;  /* privada por ser local ao bloco paralelo */
        unsigned int seed = semente_da_thread();

        #pragma omp for
        for (long i = 0; i < n_pontos; i++) {
            double x = random_double(&seed) * 2.0 - 1.0;
            double y = random_double(&seed) * 2.0 - 1.0;
            if (x * x + y * y <= 1.0) dentro_local++;
        }

        #pragma omp critical
        dentro_total += dentro_local; /* 1 secao critica por THREAD, nao por iteracao */
    }

    return 4.0 * (double)dentro_total / (double)n_pontos;
}

/* ---------- Versao 4: abordagem idiomatica do OpenMP -> reduction ---------- */
double pi_paralelo_reduction(long n_pontos) {
    long dentro = 0;

    #pragma omp parallel default(none) shared(n_pontos) reduction(+:dentro)
    {
        unsigned int seed = semente_da_thread();

        #pragma omp for
        for (long i = 0; i < n_pontos; i++) {
            double x = random_double(&seed) * 2.0 - 1.0;
            double y = random_double(&seed) * 2.0 - 1.0;
            if (x * x + y * y <= 1.0) dentro++;
        }
    }
    return 4.0 * (double)dentro / (double)n_pontos;
}

/* =====================================================================
   Demonstracao didatica isolada de private / firstprivate / lastprivate
   / shared, para observar o efeito de cada clausula separadamente.
   ===================================================================== */
void demonstrar_clausulas(void) {
    int a = 10, b = 10, c = 10, d = 0;

    printf("Antes do laco:  a=%d  b=%d  c=%d  d=%d\n", a, b, c, d);

    #pragma omp parallel for default(none) private(a) firstprivate(b) lastprivate(c) shared(d)
    for (int i = 0; i < 8; i++) {
        a = i * 10;      /* copia PRIVADA por thread; comeca com valor indefinido */
        b += i;          /* copia privada, mas INICIALIZADA com o valor de fora (10) */
        c = i;           /* ao sair do laco, 'c' externo recebe o valor calculado */
                         /* pela iteracao logicamente ULTIMA (i = 7)               */
        #pragma omp atomic
        d += 1;          /* variavel COMPARTILHADA; precisa de atomic/critical    */
    }

    printf("Depois do laco: a=%d (indefinido/nao usar) "
           "b=%d (inalterado fora, era firstprivate) "
           "c=%d (deve ser 7, ultima iteracao) "
           "d=%d (deve ser 8, soma protegida)\n\n", a, b, c, d);
}

int main(void) {
    long tamanhos[] = {1000, 10000, 100000, 1000000, 10000000};
    int n = sizeof(tamanhos) / sizeof(tamanhos[0]);

    double t0, t1;
    double tempo_erro, tempo_critical, tempo_otimizado, tempo_reduction;

    srand((unsigned int)time(NULL));

    printf("Threads disponiveis (omp_get_max_threads): %d\n\n", omp_get_max_threads());

    demonstrar_clausulas();

    printf("PI_REAL = %.10f\n\n", PI_REAL);

    for (int idx = 0; idx < n; idx++) {
        long k = tamanhos[idx];

        /* ---------- Versao Ingenua ---------- */
        t0 = omp_get_wtime();
        double pi1 = pi_paralelo_com_erro(k);
        t1 = omp_get_wtime();
        tempo_erro = t1 - t0;

        /* ---------- Versao Critical ---------- */
        t0 = omp_get_wtime();
        double pi2 = pi_paralelo_critical(k);
        t1 = omp_get_wtime();
        tempo_critical = t1 - t0;

        /* ---------- Versao Otimizada ---------- */
        t0 = omp_get_wtime();
        double pi3 = pi_paralelo_otimizado(k);
        t1 = omp_get_wtime();
        tempo_otimizado = t1 - t0;

        /* ---------- Versao Reduction ---------- */
        t0 = omp_get_wtime();
        double pi4 = pi_paralelo_reduction(k);
        t1 = omp_get_wtime();
        tempo_reduction = t1 - t0;

        /* ---------- Impressao dos resultados ---------- */
        printf("\n===============================================================\n");
        printf("Quantidade de Pontos: %ld\n", k);
        printf("===============================================================\n");
        printf("%-15s %-15s %-12s\n", "Versao", "Pi Estimado", "Tempo (s)");
        printf("---------------------------------------------------------------\n");
        printf("%-15s %-15.8f %-12.6f\n", "Ingenua",    pi1, tempo_erro);
        printf("%-15s %-15.8f %-12.6f\n", "Critical",   pi2, tempo_critical);
        printf("%-15s %-15.8f %-12.6f\n", "Otimizada",  pi3, tempo_otimizado);
        printf("%-15s %-15.8f %-12.6f\n", "Reduction",  pi4, tempo_reduction);
    }

    return 0;
}