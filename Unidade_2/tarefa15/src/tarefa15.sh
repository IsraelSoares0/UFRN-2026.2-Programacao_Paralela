#!/bin/bash
#SBATCH --job-name=barra1d
#SBATCH --time=0-1:0

#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive

#SBATCH --mail-user=israel.soares.118@ufrn.edu.br
#SBATCH --mail-type=ALL

EXE=./tarefa15
RESULTADOS=resultados_todos.csv
TAMANHOS="1000 5000 10000"   # N_PONTOS (o tempo cresce ~N^3, cuidado com valores grandes)

# Pegamos os dois primeiros nos alocados
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}
N1=${NODES[1]}
echo "Nos: $N0 $N1"

mkdir -p dados
echo "caso,versao,processos,pontos,tempo_s" > "$RESULTADOS"

roda_caso() {
    local rotulo=$1 pontos=$2
    shift 2
    echo "=== $rotulo | pontos=$pontos ==="
    rm -f dados/tempos.csv
    srun "$@" "$EXE" "$pontos"
    tail -n +2 dados/tempos.csv | sed "s/^/$rotulo,/" >> "$RESULTADOS"
}

for N in $TAMANHOS; do
    # --- INTER-NO: processos distribuidos em 2 nos (comunicacao pela rede) ---
    roda_caso "inter_2p" "$N" \
        --nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist="$N0,$N1"
    roda_caso "inter_4p" "$N" \
        --nodes=2 --ntasks=4 --ntasks-per-node=2 --nodelist="$N0,$N1"
    roda_caso "inter_8p" "$N" \
        --nodes=2 --ntasks=8 --ntasks-per-node=4 --nodelist="$N0,$N1"

done

echo "Concluido. Resumo em $RESULTADOS"