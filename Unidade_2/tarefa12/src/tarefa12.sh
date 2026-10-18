#!/bin/bash
#SBATCH --job-name=tarefa12
#SBATCH --time=0-6:0

#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

#SBATCH --mail-user=israel.soares.118@ufrn.edu.br
#SBATCH --mail-type=ALL

set -u

# ----------------------------- CONFIGURACAO ---------------------------
declare -A EXE=(
    [1]=${EXE1:-./versao1}
    [2]=${EXE2:-./versao2}
    [3]=${EXE3:-./versao3}
    [4]=${EXE4:-./versao4}
)

VERSOES=${VERSOES:-"1 2 3 4"}
EXPERIMENTOS=${EXPERIMENTOS:-"tf n"}
NS_VERSOES=${NS_VERSOES:-"3 4"}   # versoes cujo executavel aceita N (2o arg)

# Nucleos (do menor para o maior; o primeiro deve ser 1 p/ escalabilidade forte)
CORES=${CORES:-"1,2,4,8,16,32,64"}
REPS=${REPS:-3}

# Entradas (do menor para o maior problema).
# tf: o trabalho dobra a cada entrada (passos de tempo proporcionais a t_final).
TFS=${TFS:-"0.005,0.01,0.02,0.04,0.08,0.16"}
# n : N cresce por sqrt(2); como o trabalho cresce com N^4, ele quadruplica
#     a cada entrada. Reduza a lista se ficar pesado.
NS=${NS:-"256,362,512,724,1024,1448"}
TF_FIXO=${TF_FIXO:-0.01}          # t_final usado no experimento "n"

# Politicas de OMP_SCHEDULE testadas na versao 1 (separadas por espaco)
SCHEDS_V1=${SCHEDS_V1:-"static dynamic,1 dynamic,16 guided"}

PASCAL=${PASCAL:-pascalanalyzer}
PIN=${PIN:-1}                     # 1 = OMP_PROC_BIND=close, OMP_PLACES=cores
RETOMAR=${RETOMAR:-1}
DRY=${DRY:-0}
TESTE=${TESTE:-0}

WRAP_DIR=${WRAP_DIR:-wrappers}
REL_DIR=${REL_DIR:-relatorios}
LOG_DIR=${LOG_DIR:-logs}

# Mini-execucao de verificacao
if [[ $TESTE == 1 ]]; then
    CORES="1,2"; REPS=1
    TFS="0.001,0.002"; NS="128,181"; TF_FIXO=0.001
    SCHEDS_V1="static dynamic,16"
    REL_DIR=relatorios_teste; LOG_DIR=logs_teste
fi
# ----------------------------------------------------------------------

executar() {                      # respeita DRY
    if [[ $DRY == 1 ]]; then echo "[dry] $*"; else "$@"; fi
}

suporta_n() {                     # a versao aceita N como 2o argumento?
    [[ " $NS_VERSOES " == *" $1 "* ]]
}

verificar_exe() {                 # o executavel existe e e executavel?
    local v=$1 e=${EXE[$1]}
    if [[ -x $e ]]; then return 0; fi
    echo "ERRO: executavel da versao $v nao encontrado: $e (use EXE$v=caminho)" >&2
    [[ $DRY == 1 ]] && return 0
    return 1
}

# Wrapper do experimento "n": o pascalanalyzer passa UMA entrada por vez;
# o wrapper completa com o t_final fixo:  ns_vX <t_final> <N>
criar_wrapper_n() {
    local v=$1 w="$WRAP_DIR/wrap_n_v$v.sh" e=${EXE[$1]}
    [[ $DRY == 1 ]] && { echo "[dry] criar $w"; return; }
    e="$(cd "$(dirname "$e")" && pwd)/$(basename "$e")"
    printf '#!/bin/bash\nexec "%s" "%s" "$1"\n' "$e" "$TF_FIXO" > "$w"
    chmod +x "$w"
}

FALHAS=()
rodar() {                         # rodar <exp> <versao> <schedule> <rotulo>
    local exp=$1 v=$2 sched=$3 rot=$4
    local tag="${exp}_v${v}_${rot}"
    local out="$REL_DIR/${tag}.json" app inputs

    if [[ $exp == tf ]]; then app=${EXE[$v]}; inputs=$TFS
    else                      app="$WRAP_DIR/wrap_n_v$v.sh"; inputs=$NS; fi

    if [[ $RETOMAR == 1 && -s $out ]]; then
        echo ">> [$tag] ja existe, pulando"; return
    fi
    echo ">> [$tag] OMP_SCHEDULE=$sched | cores=$CORES | entradas=$inputs | reps=$REPS"
    if [[ $DRY == 1 ]]; then
        echo "[dry] env OMP_SCHEDULE=$sched $PASCAL $app -t man -c $CORES -i $inputs -r $REPS -o $out"
    else
        env OMP_SCHEDULE="$sched" "$PASCAL" "$app" \
            -t man -c "$CORES" -i "$inputs" -r "$REPS" -o "$out" \
            2>&1 | tee "$LOG_DIR/$tag.log"
    fi
    if [[ $DRY != 1 && ! -s $out ]]; then
        echo "ERRO: [$tag] nao gerou $out (veja $LOG_DIR/$tag.log)" >&2
        FALHAS+=("$tag")
    fi
}

# ------------------------------- PRINCIPAL ----------------------------
if [[ $DRY != 1 ]] && ! command -v "$PASCAL" >/dev/null 2>&1; then
    echo "ERRO: '$PASCAL' nao encontrado no PATH (carregue o modulo/venv do PaScal)." >&2
    exit 1
fi

for v in $VERSOES; do verificar_exe "$v" || exit 1; done

executar mkdir -p "$WRAP_DIR" "$REL_DIR" "$LOG_DIR"

if [[ $PIN == 1 ]]; then
    export OMP_PROC_BIND=close OMP_PLACES=cores
fi

maxcores=${CORES##*,}
echo "Cores visiveis: $(nproc) | maior contagem pedida: $maxcores"
if (( $(nproc) < maxcores )); then
    echo "AVISO: ha menos cores visiveis que o pedido em CORES; ajuste CORES." >&2
fi

for exp in $EXPERIMENTOS; do
    for v in $VERSOES; do
        if [[ $exp == n ]]; then
            if ! suporta_n "$v"; then
                echo "AVISO: versao $v nao aceita N como argumento; experimento 'n' pulado."
                continue
            fi
            criar_wrapper_n "$v"
        fi
        if [[ $v == 1 ]]; then
            for s in $SCHEDS_V1; do
                rodar "$exp" "$v" "$s" "${s//,/-}"
            done
        else
            rodar "$exp" "$v" static "static-fixo"
        fi
    done
done

echo
echo "=== Relatorios gerados em $REL_DIR ==="
[[ $DRY == 1 ]] || ls -lh "$REL_DIR"/*.json 2>/dev/null
if (( ${#FALHAS[@]} )); then
    echo "Falharam: ${FALHAS[*]}" >&2
    exit 1
fi
echo "Concluido. Envie os .json ao PaScal Viewer para comparar os heatmaps."