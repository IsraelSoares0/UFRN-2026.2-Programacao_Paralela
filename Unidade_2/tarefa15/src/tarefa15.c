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
 * Executar: ./barra_1d
 *
 * Saída: arquivo "temperaturas.csv" com a temperatura ao longo do anel
 *        em diferentes instantes de tempo (uma linha por instante salvo).
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/stat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Parâmetros de simulação */
#define N_PONTOS 1000           /* Número de pontos na barra */
#define COMPRIMENTO 1.0         /* Comprimento da barra (m) */
#define ALPHA 0.01              /* Difusividade térmica (m²/s) */
#define T_TOTAL 2.0             /* Tempo total de simulação (s) */
#define TEMP_BASE 0.0           /* temperatura nas bordas/base (ºC) */
#define TEMP_CENTRO 100.0       /* temperatura no centro da barra (ºC) */
#define N_SALVAS 10             /* quantos instantes de tempo salvar no CSV */

#define TAG_DIREITA 0
#define TAG_ESQUERDA 1
#define BLOCO 100               /* celulas calculadas entre chamdas de MPI_Teste (V3) */

typedef struct {
    int rank, size;
    int esquerda, direita;      /* vizinhos no anel */
    int n_total;                /* pontos na barra inteira */
    int n;                      /* pontos no processo atual */
    int offset;                 /* indice global da 1 celula real */
    double dx, dt, r;
    int n_passos, intervalo_salvar;
    int *counts, *displs;       /* usados no Gatherv (so rank 0) */
    double *global;             /* buffer da barra inteira (so rank 0) */
} Sim;

/**
 * FUNCOES AUXILIARES
 */

static void atualiza(const double *u, double *un, int ini, int fim, double r)
{
    for (int i = ini; i <= fim; i++)
        un[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
}

static void condicao_inicial(const Sim *s, double *u)
{
    for (int i = 1; i < s->n; i++) {
        double x = (s->offset + i - 1) * s->dx;
        double fase = 2.0 * M_PI * (x - COMPRIMENTO / 2.0) / COMPRIMENTO;
        u[i] = TEMP_BASE + (TEMP_CENTRO - TEMP_BASE) * (cos(fase) + 1.0) / 2.0;
    }
}

static void salvar_linha(const Sim *s, FILE *f, double t, const double *u)
{
    MPI_Gatherv(&u[1], s->n, MPI_DOUBLE, s->global, s->counts, s->displs,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (s->rank == 0) {
        fprintf(f, "%.6f", t);
        for (int i = 0; i < s->n_total; i++) fprintf(f, ",%.6f", s->global[i]);
        fprintf(f, "\n");
    }
}

static FILE *abrir_csv(const Sim *s, const char *nome)
{
    if (s->rank != 0) return NULL;
    FILE *f = fopen(nome, "w");

    if (!f) {
        fprintf(stderr, "Erro ao criar %s\n", nome);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(f, "tempo");
    for (int i = 0; i < s->n_total; i++) fprintf(f, ",%.6f", i * s->dx);
    fprintf(f, "\n");

    return f;
}

typedef void (*PassoFn)(const Sim *, double *, double *);

static double executar(const Sim *s, const char *arquivo, PassoFn passo_fn)
{
    double *u = calloc(s->n + 2, sizeof(double));
    double *un = calloc(s->n + 2, sizeof(double));

    condicao_inicial(s, u);
    FILE *f = abrir_csv(s, arquivo);
    salvar_linha(s, f, 0.0, u);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_acum = 0.0;

    for (int passo = 1; passo <= s->n_passos; passo++) {
        double t0 = MPI_Wtime();
        passo_fn(s, u, un);
        double *tmp = u; u = un; un = tmp;
        t_acum += MPI_Wtime() - t0;

        if (passo % s->intervalo_salvar == 0 || passo == s->n_passos)
            salvar_linha(s, f, passo * s->dt, u);
    }

    if (f) fclose(f);
    free(u);
    free(un);

    double t_max;
    MPI_Reduce(&t_acum, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return t_max;
}

/**
 * V1: SEND/RECV BLOQUEANTES
 */
static void passo_v1(const Sim *s, double *u, double *un)
{
    int n = s->n;
    const int ultimo = s->size - 1;

    /* Fase 1: dados indo para direita (u[n] -> ghost u[0])*/
    if (s->rank == 0) {
        MPI_Send(&u[n], 1, MPI_DOUBLE, s->direita, TAG_DIREITA, MPI_COMM_WORLD);
        MPI_Recv(&u[0], 1, MPI_DOUBLE, s->esquerda, TAG_DIREITA, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&u[0], 1, MPI_DOUBLE, s->esquerda, TAG_DIREITA, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(&u[n], 1, MPI_DOUBLE, s->direita, TAG_DIREITA, MPI_COMM_WORLD);
    }

    /* Fase 2: dados indo para esquerda (u[1] -> ghost u[n+1])*/
    if (s->rank == ultimo) {
        MPI_Send(&u[1], 1, MPI_DOUBLE, s->esquerda, TAG_ESQUERDA, MPI_COMM_WORLD);
        MPI_Recv(&u[n + 1], 1, MPI_DOUBLE, s->direita, TAG_ESQUERDA,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&u[n + 1], 1, MPI_DOUBLE, s->direita, TAG_ESQUERDA,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&u[1], 1, MPI_DOUBLE, s->esquerda, TAG_ESQUERDA, MPI_COMM_WORLD);
    }

    atualiza(u, un, 1, n, s->r);
}

/**
 * V2: COMUNICACAO NAO BLOQUEANTE, ESPERA ANTES DE CALCULAR
 */
static void passo_v2(const Sim *s, double *u, double *un) {
    int n = s->n;
    MPI_Request req[4];
 
    MPI_Irecv(&u[0], 1, MPI_DOUBLE, s->esquerda, TAG_DIREITA, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(&u[n + 1], 1, MPI_DOUBLE, s->direita, TAG_ESQUERDA, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(&u[n], 1, MPI_DOUBLE, s->direita, TAG_DIREITA, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(&u[1], 1, MPI_DOUBLE, s->esquerda, TAG_ESQUERDA, MPI_COMM_WORLD, &req[3]);
 
    for (int k = 0; k < 4; k++) MPI_Wait(&req[k], MPI_STATUS_IGNORE);
 
    atualiza(u, un, 1, n, s->r);
}

/**
 * V3: SOBREPOSICAO
 */
static void passo_v3(const Sim *s, double *u, double *un) {
    int n = s->n;
    MPI_Request req[4];
    int flag;
 
    MPI_Irecv(&u[0], 1, MPI_DOUBLE, s->esquerda, TAG_DIREITA, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(&u[n + 1], 1, MPI_DOUBLE, s->direita, TAG_ESQUERDA, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(&u[n], 1, MPI_DOUBLE, s->direita, TAG_DIREITA, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(&u[1], 1, MPI_DOUBLE, s->esquerda, TAG_ESQUERDA, MPI_COMM_WORLD, &req[3]);
 
    /* Pontos internos, em blocos, testando a comunicação entre eles */
    for (int ini = 2; ini <= n - 1; ini += BLOCO) {
        int fim = ini + BLOCO - 1;
        if (fim > n - 1) fim = n - 1;
        atualiza(u, un, ini, fim, s->r);
        for (int k = 0; k < 4; k++) MPI_Test(&req[k], &flag, MPI_STATUS_IGNORE);
    }
 
    /* Garante o término da comunicação antes de usar o halo */
    for (int k = 0; k < 4; k++) MPI_Wait(&req[k], MPI_STATUS_IGNORE);
 
    /* Pontos de borda (dependem das células fantasmas) */
    atualiza(u, un, 1, 1, s->r);
    atualiza(u, un, n, n, s->r);
}

/**
 * FUNCOES PARA CADA VERSAO
 */

double versao1(const Sim *s) { return executar(s, "dados/temperaturas_v1.csv", passo_v1); }
double versao2(const Sim *s) { return executar(s, "dados/temperaturas_v2.csv", passo_v2); }
double versao3(const Sim *s) { return executar(s, "dados/temperaturas_v3.csv", passo_v3); }

/**
 * MAIN
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
 
    Sim s;
    MPI_Comm_rank(MPI_COMM_WORLD, &s.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s.size);
 
    s.n_total = (argc > 1) ? atoi(argv[1]) : N_PONTOS_PADRAO;
 
    if (s.size < 2 || s.n_total < 2 * s.size) {
        if (s.rank == 0)
            fprintf(stderr, "Use pelo menos 2 processos e N_PONTOS >= 2*processos.\n");
        MPI_Finalize();
        return 1;
    }
 
    /* Vizinhos no anel */
    s.esquerda = (s.rank - 1 + s.size) % s.size;
    s.direita = (s.rank + 1) % s.size;
 
    /* Divisão da barra (o resto vai para os primeiros ranks) */
    int base = s.n_total / s.size, resto = s.n_total % s.size;
    s.n = base + (s.rank < resto ? 1 : 0);
    s.offset = s.rank * base + (s.rank < resto ? s.rank : resto);
 
    /* Parâmetros numéricos */
    s.dx = COMPRIMENTO / s.n_total;
    double dt_max = 0.5 * s.dx * s.dx / ALPHA;
    s.dt = 0.9 * dt_max;
    s.n_passos = (int)ceil(T_TOTAL / s.dt);
    s.dt = T_TOTAL / s.n_passos;
    s.r = ALPHA * s.dt / (s.dx * s.dx);
    s.intervalo_salvar = s.n_passos / N_SALVAS;
    if (s.intervalo_salvar < 1) s.intervalo_salvar = 1;
 
    /* Buffers do Gatherv (só no rank 0) */
    s.counts = s.displs = NULL;
    s.global = NULL;
    if (s.rank == 0) {
        mkdir("dados", 0777);
        s.counts = malloc(s.size * sizeof(int));
        s.displs = malloc(s.size * sizeof(int));
        s.global = malloc(s.n_total * sizeof(double));
        for (int p = 0, d = 0; p < s.size; p++) {
            s.counts[p] = base + (p < resto ? 1 : 0);
            s.displs[p] = d;
            d += s.counts[p];
        }
        printf("Processos: %d | Pontos: %d | Passos: %d | r = %.4f\n",
               s.size, s.n_total, s.n_passos, s.r);
    }
    MPI_Barrier(MPI_COMM_WORLD);
 
    double t1 = versao1(&s);
    double t2 = versao2(&s);
    double t3 = versao3(&s);
 
    if (s.rank == 0) {
        printf("V1 (Send/Recv)       : %.6f s\n", t1);
        printf("V2 (Isend/Irecv/Wait): %.6f s\n", t2);
        printf("V3 (sobreposicao)    : %.6f s\n", t3);
 
        FILE *ft = fopen("dados/tempos.csv", "r");
        int novo = (ft == NULL);
        if (ft) fclose(ft);
        ft = fopen("dados/tempos.csv", "a");
        if (ft) {
            if (novo) fprintf(ft, "versao,processos,pontos,tempo_s\n");
            fprintf(ft, "V1,%d,%d,%.6f\n", s.size, s.n_total, t1);
            fprintf(ft, "V2,%d,%d,%.6f\n", s.size, s.n_total, t2);
            fprintf(ft, "V3,%d,%d,%.6f\n", s.size, s.n_total, t3);
            fclose(ft);
        }
        free(s.counts);
        free(s.displs);
        free(s.global);
    }
 
    MPI_Finalize();
    return 0;
}