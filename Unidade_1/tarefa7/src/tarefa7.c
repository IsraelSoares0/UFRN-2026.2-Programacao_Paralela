/*
 * Exemplo: lista encadeada + OpenMP tasks
 *
 * Compilar: gcc -fopenmp -Wall -o lista_tasks lista_tasks.c
 * Executar: ./lista_tasks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef struct Node {
    char nome_arquivo[64];
    struct Node *prox;
} Node;

/* Cria um novo nó com o nome do arquivo informado */
Node *criar_no(const char *nome) {
    Node *n = (Node *) malloc(sizeof(Node));
    if (!n) {
        fprintf(stderr, "Erro de alocação de memória\n");
        exit(EXIT_FAILURE);
    }
    strncpy(n->nome_arquivo, nome, sizeof(n->nome_arquivo) - 1);
    n->nome_arquivo[sizeof(n->nome_arquivo) - 1] = '\0';
    n->prox = NULL;
    return n;
}

/* Insere um novo nó no final da lista */
void inserir(Node **cabeca, const char *nome) {
    Node *novo = criar_no(nome);
    if (*cabeca == NULL) {
        *cabeca = novo;
        return;
    }
    Node *atual = *cabeca;
    while (atual->prox != NULL) {
        atual = atual->prox;
    }
    atual->prox = novo;
}

/* Libera toda a memória da lista */
void liberar_lista(Node *cabeca) {
    Node *tmp;
    while (cabeca != NULL) {
        tmp = cabeca;
        cabeca = cabeca->prox;
        free(tmp);
    }
}

/* "Processa" um nó */
void processar_arquivo(const char *nome_arquivo) {
    printf("Arquivo '%s' processado pela thread %d\n",
           nome_arquivo, omp_get_thread_num());
}

int main(void) {
    Node *lista = NULL;
    char nome[32];
    int total_nos = 10;

    /* Monta a lista encadeada */
    for (int i = 1; i <= total_nos; i++) {
        snprintf(nome, sizeof(nome), "arquivo_%02d.dat", i);
        inserir(&lista, nome);
    }

    #pragma omp parallel
    {
        /* IMPORTANTE:
         * "single" garante que APENAS UMA thread percorre a lista e
         * cria as tasks. Sem isso, todas as threads da região paralela
         * executariam o laço, cada uma criando uma task para CADA nó,
         * resultando em processamento duplicado (N vezes, sendo N o
         * número de threads).
         *
         * "nowait" evita uma barreira implícita ao final do single,
         * permitindo que as threads ociosas comecem a executar as
         * tasks já criadas assim que possível.
         */
        #pragma omp single nowait
        {
            Node *atual = lista;
            while (atual != NULL) {
                Node *node_atual = atual; /* cópia local para a task */

                #pragma omp task firstprivate(node_atual)
                {
                    processar_arquivo(node_atual->nome_arquivo);
                }

                atual = atual->prox;
            }
        }
    }

    liberar_lista(lista);
    return 0;
}