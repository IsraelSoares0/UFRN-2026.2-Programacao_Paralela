/*
 * duas_listas.c
 *
 * Compilar:  gcc -fopenmp -Wall -O2 duas_listas.c -o duas_listas
 * Executar:  ./duas_listas [N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
    int N = (argc > 1) ? atoi(argv[1]) : 20;

    Node *lista1 = NULL;
    Node *lista2 = NULL;

    printf("Realizando %d insercoes distribuidas em 2 listas...\n", N);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                #pragma omp task firstprivate(i)
                {
                    unsigned int seed = semente_da_thread() ^ (unsigned int)(i * 2246822519u);

                    int escolha = (int)(xorshift32(&seed) % 2u);
                    int valor = (int)(xorshift32(&seed) % 1000u);

                    if (escolha == 0) {
                        #pragma omp critical(lista1_critico)
                        {
                            insere_lista(&lista1, valor);
                        }
                    } else {
                        #pragma omp critical(lista2_critico)
                        {
                            insere_lista(&lista2, valor);
                        }
                    }

                }
            }
        } /* Fim do Single */
    }

    printf("Lista 1: %d elementos\n", tamanho_lista(lista1));
    printf("Lista 2: %d elementos\n", tamanho_lista(lista2));
 
    libera_lista(lista1);
    libera_lista(lista2);
 
    return 0;
}