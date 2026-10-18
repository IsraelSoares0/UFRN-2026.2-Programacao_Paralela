#!/bin/bash
#SBATCH --job-name=tarefa18
#SBATCH --partition=gpu-4-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:0

set -e

cd "$SLURM_SUBMIT_DIR"

# ---------- Execução ----------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Nó: $(hostname)"
nvidia-smi -L || true

# Argumentos: N máximo e número de repetições
./tarefa18 200000000 5

echo "Resultados em: $SLURM_SUBMIT_DIR/resultados.csv"