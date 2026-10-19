#!/bin/bash
#SBATCH --job-name=tarefa13
#SBATCH --time=0-4:0

#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --exclusive

#SBATCH --mail-user=israel.soares.118@ufrn.edu.br
#SBATCH --mail-type=ALL

# Tarefa 13: escalabilidade do Navier-Stokes (v4) com diferentes afinidades de threads.
# Use o MESMO no (particao) da tarefa 12: ajuste --partition acima se necessario.

SRC=tarefa13.c
EXE=./tarefa13
N=2048                 # malha N x N
TF=0.001               # tempo final simulado (define o numero de passos)
REPS=5                 # repeticoes por configuracao (a variacao entre repeticoes chegou a ~30%)
TIMEOUT=180            # limite por execucao (s); casos patologicos (ex.: master) sao abortados
BIND_ARG=env           # env = usa OMP_PROC_BIND do ambiente | master|close|spread = clausula proc_bind

RESULTADOS=resultados_t13.csv
RESUMO=resumo_t13.csv
LOGDIR=logs_t13

# Configuracoes: rotulo|OMP_PROC_BIND|OMP_PLACES  (places vazio = padrao do runtime)
CONFIGS=(
    "sem_afinidade|false|"
    "true_cores|true|cores"
    "close_threads|close|threads"
    "close_cores|close|cores"
    "close_sockets|close|sockets"
    "spread_threads|spread|threads"
    "spread_cores|spread|cores"
    "spread_sockets|spread|sockets"
    "master_cores|master|cores"
    # Exemplo de lugares explicitos (ajustar a topologia do no, veja lscpu):
    # "spread_manual|spread|{0:64:1},{64:64:1}"
)

# --- Compila se o executavel nao existir ---
if [[ ! -x "$EXE" ]]; then
    echo "Compilando $SRC ..."
    gcc -O3 -Wall -fopenmp -o "${EXE#./}" "$SRC" -lm -lmpascalops || { echo "Falha na compilacao"; exit 1; }
fi

# --- Diagnostico da topologia do no ---
echo "=== No: $(hostname) ==="
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA'
echo

# --- CPUs realmente visiveis para cada execucao ---
# ATENCAO: com --ntasks=1 o Slurm prende o proprio script (e o nproc dele) a 1 CPU.
# Por isso cada execucao roda via srun com todas as CPUs do no e sem cpu-bind;
# o numero de CPUs e medido dentro de um passo do srun, nao no shell do job.
NCPUS_ALLOC=${SLURM_CPUS_ON_NODE:-$(nproc --all)}
SRUN_OPTS=(--ntasks=1 --nodes=1 --cpus-per-task="$NCPUS_ALLOC" --cpu-bind=none --mem=0)
NCPUS=$(srun "${SRUN_OPTS[@]}" nproc)
echo "CPUs alocadas: $NCPUS_ALLOC | CPUs visiveis no passo srun: $NCPUS"
if (( NCPUS < 2 )); then
    echo "ERRO: apenas $NCPUS CPU visivel; o estudo de escalabilidade nao faz sentido. Abortando."
    exit 1
fi

# --- Numeros de threads: potencias de 2 ate o total de CPUs ---
THREADS=()
for ((t = 1; t < NCPUS; t *= 2)); do THREADS+=("$t"); done
THREADS+=("$NCPUS")
echo "Threads: ${THREADS[*]}"

# Limpa variaveis que poderiam interferir
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY OMP_NUM_THREADS

mkdir -p "$LOGDIR"
echo "caso,rep,N,threads,proc_bind,passos,tempo_calculo_s,tempo_total_s,erro_max,migracoes" > "$RESULTADOS"

roda() {
    local rotulo=$1 bind=$2 places=$3 t=$4 rep=$5
    local log="$LOGDIR/${rotulo}_T${t}_r${rep}.log"
    local amb=("OMP_NUM_THREADS=$t" "OMP_PROC_BIND=$bind")
    [[ -n "$places" ]] && amb+=("OMP_PLACES=$places")

    timeout "$TIMEOUT" srun "${SRUN_OPTS[@]}" \
        env -u OMP_PLACES -u OMP_PROC_BIND "${amb[@]}" \
        "$EXE" "$TF" "$N" "$BIND_ARG" > "$log" 2>&1
    local status=$?

    if grep -q '^RESULT' "$log"; then
        grep '^RESULT' "$log" | sed "s/^RESULT,/$rotulo,$rep,/" >> "$RESULTADOS"
    else
        echo "  [FALHA status=$status] $rotulo T=$t rep=$rep (veja $log)"
    fi
}

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r rotulo bind places <<< "$cfg"
    for t in "${THREADS[@]}"; do
        echo "=== $rotulo | OMP_PROC_BIND=$bind OMP_PLACES=${places:-padrao} | threads=$t ==="
        for ((rep = 1; rep <= REPS; rep++)); do
            roda "$rotulo" "$bind" "$places" "$t" "$rep"
        done
    done
done

# --- Resumo: medias, speedup e eficiencia (base: sem_afinidade com 1 thread) ---
awk -F, 'NR == 1 { next }
{
    k = $1 "," $4
    sc[k] += $7; st[k] += $8; sm[k] += $10; cnt[k]++
    if (!(k in ord)) { ord[k] = ++n; chave[n] = k }
}
END {
    b = "sem_afinidade,1"
    base = (cnt[b] > 0) ? sc[b] / cnt[b] : 0
    print "caso,threads,tempo_calculo_medio_s,tempo_total_medio_s,speedup,eficiencia,migracoes_medias"
    for (i = 1; i <= n; i++) {
        k = chave[i]; split(k, a, ",")
        tm = sc[k] / cnt[k]
        sp = (base > 0) ? base / tm : 0
        printf "%s,%s,%.4f,%.4f,%.3f,%.3f,%.1f\n", a[1], a[2], tm, st[k] / cnt[k], sp, sp / a[2], sm[k] / cnt[k]
    }
}' "$RESULTADOS" > "$RESUMO"

echo
echo "Concluido. Dados brutos: $RESULTADOS | resumo: $RESUMO | logs: $LOGDIR/"
column -s, -t "$RESUMO" 2>/dev/null || cat "$RESUMO"