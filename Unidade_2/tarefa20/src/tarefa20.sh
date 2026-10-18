#!/bin/bash
#SBATCH --job-name=tarefa20
#SBATCH --time=0-1:0

#SBATCH --partition=gpu-8-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH --mail-user=israel.soares.118@ufrn.edu.br
#SBATCH --mail-type=ALL

EXE=./tarefa20
RESULTADOS=resultados_tarefa20.csv
PERFIS=perfis_nsys_t20

TAMANHOS="500 1000 2000 4000 8000"   # grade n x n
NSTEPS=100                           # passos de tempo (mantem r estavel)
REPS=3                               # repeticoes por medicao (sem nsys)

echo "=== GPU alocada ==="
nvidia-smi
which nsys || echo "AVISO: nsys nao encontrado (tente: module load cuda / nvhpc)"

mkdir -p "$PERFIS"

# ---------- 1) Medicao de tempo (sem profiler) ----------
echo "n,nsteps,rep,tempo_solve_s,tempo_total_s,erro_l2" > "$RESULTADOS"

for N in $TAMANHOS; do
    for R in $(seq 1 $REPS); do
        echo "=== Medicao | n=$N | rep=$R ==="
        SAIDA=$(srun --mem=0 "$EXE" "$N" "$NSTEPS")
        SOLVE=$(echo "$SAIDA" | awk '/Solve time/ {print $4}')
        TOTAL=$(echo "$SAIDA" | awk '/Total time/ {print $4}')
        ERRO=$(echo "$SAIDA"  | awk '/Error \(L2norm\)/ {print $3}')
        echo "$N,$NSTEPS,$R,$SOLVE,$TOTAL,$ERRO" >> "$RESULTADOS"
    done
done

# ---------- 2) Perfilamento com nsys ----------
for N in $TAMANHOS; do
    echo "=== nsys | n=$N ==="
    srun --mem=0 nsys profile \
        --trace=cuda,openmp,nvtx,osrt \
        --stats=true \
        --force-overwrite=true \
        -o "$PERFIS/heat_n${N}" \
        "$EXE" "$N" "$NSTEPS" > "$PERFIS/heat_n${N}_stats.txt" 2>&1
done

echo "Concluido."
echo "Tempos:  $RESULTADOS"
echo "Perfis:  $PERFIS/*.nsys-rep  e  $PERFIS/*_stats.txt"