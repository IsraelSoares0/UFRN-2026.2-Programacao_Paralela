#!/bin/bash
#SBATCH --job-name=pingpong
#SBATCH --time=0-1:0
#SBATCH --output=pingpong_%j.out
#SBATCH --error=pingpong_%j.err

#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive

# Pegamos os dois primeiros nos alocados
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}
N1=${NODES[1]}

echo "Nos: $N0 $N1"

# --- Caso 1: INTRA-NO (2 processos no mesmo no, memoria compartilhada) ---
srun --nodes=1 --ntasks=2 --nodelist="$N0" ./tarefa14 resultados_intra.csv

# --- Caso 2: INTER-NO (1 processo por no, comunicacao pela rede) ---
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist="$N0,$N1" \
     ./tarefa14 resultados_inter.csv

echo "Concluido."
