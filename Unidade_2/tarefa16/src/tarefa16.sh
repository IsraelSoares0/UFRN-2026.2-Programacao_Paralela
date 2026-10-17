#!/bin/bash
#SBATCH --job-name=tarefa16
#SBATCH --time=0-1:0

#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive

#SBATCH --mail-user=israel.soares.118@ufrn.edu.br
#SBATCH --mail-type=ALL

EXE=./tarefa16
RESULTADOS=resultados_todos.csv
TAMANHOS="1024 2048 4096 8192 16384"    # matrizes quadradas M = N
REPS=5                                  # repeticoes por medicao (o programa imprime a media)

# Pegamos os dois primeiros nos alocados
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}
N1=${NODES[1]}
echo "Nos: $N0 $N1"

# Evita o aviso/erro do BTL openib (placa mlx5_0) no Open MPI
export OMPI_MCA_btl='^openib'

# Diagnostico: limites de memoria vistos por um passo do srun
echo "=== Limites no no $N0 ==="
srun --mem=0 --nodes=1 --ntasks=1 --nodelist="$N0" bash -c 'ulimit -a; free -m'

echo "caso,processos,M,N,tempo_total_s,tempo_calculo_s" > "$RESULTADOS"

roda_caso() {
    local rotulo=$1 tamanho=$2
    shift 2
    echo "=== $rotulo | M=N=$tamanho ==="
    srun --mem=0 "$@" "$EXE" "$tamanho" "$tamanho" "$REPS" \
        | grep -v '^#' | sed "s/^/$rotulo,/" >> "$RESULTADOS"
}

for N in $TAMANHOS; do
    # --- INTRA-NO: todos os processos no mesmo no (memoria compartilhada) ---
    roda_caso "intra_1p" "$N" \
        --nodes=1 --ntasks=1 --nodelist="$N0"
    roda_caso "intra_2p" "$N" \
        --nodes=1 --ntasks=2 --nodelist="$N0"
    roda_caso "intra_4p" "$N" \
        --nodes=1 --ntasks=4 --nodelist="$N0"
    roda_caso "intra_8p" "$N" \
        --nodes=1 --ntasks=8 --nodelist="$N0"

    # --- INTER-NO: processos distribuidos em 2 nos (comunicacao pela rede) ---
    roda_caso "inter_2p" "$N" \
        --nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist="$N0,$N1"
    roda_caso "inter_4p" "$N" \
        --nodes=2 --ntasks=4 --ntasks-per-node=2 --nodelist="$N0,$N1"
    roda_caso "inter_8p" "$N" \
        --nodes=2 --ntasks=8 --ntasks-per-node=4 --nodelist="$N0,$N1"
    roda_caso "inter_16p" "$N" \
        --nodes=2 --ntasks=16 --ntasks-per-node=8 --nodelist="$N0,$N1"
done

echo "Concluido. Resumo em $RESULTADOS"