/** TAREFA 14 - PingPong MPI
 * 
 * Compilar: mpicc -O2 -Wall -o tarefa14 tarefa14.c
 * 
 * Executar: mpirun -np 2 ./tarefa14 [saida.csv]
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN_EXP     3       /* 2^3  = 8 bytes*/
#define MAX_EXP     26      /* 2^26 = 64 MiB*/
#define WARMUP      20
#define TRIALS      5
#define MIN_REPS    20
#define MAX_REPS    20000
#define TOTAL_BYTES (256UL * 1024 * 1024)
#define TAG         0

static void pingpong(char *buf, int n, int reps, int rank)
{
    for (int i = 0; i < reps; i++) {
        if (rank == 0) {
            MPI_Send(buf, n, MPI_CHAR, 1, TAG, MPI_COMM_WORLD);
            MPI_Recv(buf, n, MPI_CHAR, 1, TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf, n, MPI_CHAR, 0, TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(buf, n, MPI_CHAR, 0, TAG, MPI_COMM_WORLD);
        }
    }
}


int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "Erro: execute com 2 processos "
                            "(mpirun -np 2).\n");
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    size_t max_bytes = (size_t)1 << MAX_EXP;
    char *buf = malloc(max_bytes);

    if (!buf) {
        fprintf(stderr, "Rank %d: falha ao alocar %zu bytes.\n", rank, max_bytes);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    /* Toca as páginas */
    memset(buf, rank + 1, max_bytes);

    FILE *out = stdout;
    if (rank == 0 && argc > 1) {
        out = fopen(argv[1], "w");
        if (!out) {
            perror("fopen");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (rank == 0) {
        fprintf(out, "tamanho_bytes,repeticoes,tempo_total_s,rtt_s,tempo_unidirecional_s,banda_MBps\n");
    }

    for (int e = MIN_EXP; e <= MAX_EXP; e++) {
        int n = 1 << e;
        int reps = (int)(TOTAL_BYTES / (size_t)n);
        
        if (reps > MAX_REPS) reps = MAX_REPS;
        if (reps < MIN_REPS) reps = MIN_REPS;

        /* AQUECIMENTO */
        pingpong(buf, n, WARMUP, rank);

        double best = 1e30;

        for (int t = 0; t < TRIALS; t++) {
            if (rank == 0) {
                for (int i = 0; i < n; i++) buf[i] = (char)((i + e) & 0x7F);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            pingpong(buf, n, reps, rank);
            double t1 = MPI_Wtime();

            if (rank == 0) {
                double dt = t1 - t0;
                if (dt < best) best = dt;

                for (int i = 0; i < n; i++) {
                    if (buf[i] != (char)((i + e) & 0x7F)) {
                        fprintf(stderr, "Erro: eco corrompido (n=%d)\n", n);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
                }
            }
        }

        if (rank == 0) {
            double rtt = best / reps;               /* round trip */
            double oneway = rtt / 2.0;              /* one way */
            double bw = (double)n / oneway / 1e6;   /* MB/s */
            fprintf(out, "%d,%d,%.9f,%.9e,%.9e,%.3f\n",
                    n, reps, best, rtt, oneway, bw);
            fflush(out);
        }
    }

    if (rank == 0 && out != stdout) fclose(out);
    free(buf);
    MPI_Finalize();
    return EXIT_SUCCESS;
}