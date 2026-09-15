/*
 * n_listas.c
 *
 * Compilar:  gcc -fopenmp -Wall -O2 n_listas.c -o n_listas
 * Executar:  ./n_listas [M] [N]
 *            M = número de listas, N = número de inserções
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <omp.h>

/* --------------------------------------------------
 * Gerador Xorshift32 (thread-safe: cada chamador guarda seu próprio 'state')
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
 
/* Geracao de double aleatorio em [0,1) */
static inline double random_double(unsigned int *state)
{
    return (double)xorshift32(state) / (double)UINT_MAX;
}
 
static inline unsigned int semente_da_thread(void)
{
    unsigned int base = (unsigned int)time(NULL) ^
                         ((unsigned int)omp_get_thread_num() * 2654435761u);
    return base | 1u;
}


typedef struct Node {
    int data;
    struct Node *next;
} Node;

/* Insere no inicio da fila -- Não é thread-safe pro si só*/
static void insere_lista(Node **head, int valor) {
    Node *novo = malloc(sizeof(Node));
    novo->data = valor;
    novo->next = *head;
    *head = novo;
}

static int tamanho_lista(Node *head) {
    int n = 0;
    while (head) { n++; head = head->next; }
    return n;
}

static void libera_lista(Node *head) {
    while(head) {
        Node *tmp = head;
        head = head->next;
        free(tmp);
    }
}

int main(int argc, char *argv[]) {
    int M = (argc > 1) ? atoi(argv[1]) : 4;
    int N = (argc > 2) ? atoi(argv[2]) : 40;

    if (M <= 0) {
        fprintf(stderr, "M precisa ser >= 1\n");
        return 1;
    }

    Node **listas = calloc((size_t)M, sizeof(Node *));
    omp_lock_t *locks = malloc((size_t)M * sizeof(omp_lock_t));

    for (int i = 0; i < M; i++) {
        omp_init_lock(&locks[i]);
    }

    printf("Realizando %d insercoes distrubuidas em %d listas...\n", N, M);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                #pragma omp task firstprivate(i)
                {
                    unsigned int seed = semente_da_thread() ^ (unsigned int)(i * 2246822519u);
 
                    int idx   = (int)(xorshift32(&seed) % (unsigned int)M); /* lista escolhida */
                    int valor = (int)(xorshift32(&seed) % 1000u);

                    omp_set_lock(&locks[idx]);
                    insere_lista(&listas[idx], valor);
                    omp_unset_lock(&locks[idx]);
                }
            }
        }
    }

    for (int i = 0; i < M; i++) {
        printf("Lista %d: %d elementos\n", i, tamanho_lista(listas[i]));
        libera_lista(listas[i]);
    }
 
    for (int i = 0; i < M; i++) {
        omp_destroy_lock(&locks[i]);
    }

    free(locks);
    free(listas);

    return 0;
}