#!/bin/bash
# Perfilamento da tarefa 19 com o nsys (rodar SEM sudo, na pasta do binario).
#
# Antes: compilar com
#   gcc -O3 -fopenmp -foffload=nvptx-none -fcf-protection=none -fno-stack-protector \
#       -o tarefa19 tarefa19.c -lm
#
# Uso: ./tarefa19.sh   (se der "Permission denied": chmod +x tarefa19.sh)
#
# Saidas em perfis/:
#   nXXXX.nsys-rep         linha do tempo (abrir com: nsys-ui perfis/nXXXX.nsys-rep)
#   nXXXX_stats.txt        tabelas do nsys (kernel, tempo e volume de memcpy, API CUDA)
#   nXXXX_saida.txt        saida do proprio programa
#
# Atencao: os tempos das execucoes perfiladas incluem overhead do profiler.
# Os tempos de desempenho vem do resultados_tarefa19.csv (./tarefa19 sem argumentos).

EXE=./tarefa19
PERFIS=perfis
TAMANHOS="500 1000 2000 4000 8000"   # grade n x n
NSTEPS=100                           # passos de tempo

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERRO: nsys nao encontrado no PATH."
    exit 1
fi
if [ ! -x "$EXE" ]; then
    echo "ERRO: $EXE nao existe ou nao e executavel (compile primeiro)."
    exit 1
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nsys --version

# O nsys procura o QdstrmImporter ao lado do proprio executavel. Se nao achar,
# o .qdstrm nao vira .nsys-rep. Localizamos o importador para converter na mao.
IMP=$(find /opt /usr -name QdstrmImporter -type f 2>/dev/null | head -1)
if [ -z "$IMP" ]; then
    IMP=$(find / -name QdstrmImporter -type f 2>/dev/null | head -1)
fi
if [ -n "$IMP" ]; then
    echo "Importador: $IMP"
else
    echo "AVISO: QdstrmImporter nao encontrado; so sera gerado o .qdstrm bruto."
fi

mkdir -p "$PERFIS"

for N in $TAMANHOS; do
    echo "=== nsys | n=$N | nsteps=$NSTEPS ==="

    nsys profile \
        --trace=cuda,osrt \
        --sample=none \
        --force-overwrite=true \
        -o "$PERFIS/n$N" \
        "$EXE" "$N" "$NSTEPS" > "$PERFIS/n${N}_saida.txt" 2>&1

    # Converte o .qdstrm se o nsys nao conseguiu gerar o .nsys-rep sozinho
    if [ ! -f "$PERFIS/n$N.nsys-rep" ] && [ -f "$PERFIS/n$N.qdstrm" ] && [ -n "$IMP" ]; then
        echo "Convertendo n$N.qdstrm -> n$N.nsys-rep"
        "$IMP" -i "$PERFIS/n$N.qdstrm" -o "$PERFIS/n$N.nsys-rep"
    fi

    if [ -f "$PERFIS/n$N.nsys-rep" ]; then
        nsys stats --force-export=true \
            --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
            "$PERFIS/n$N.nsys-rep" > "$PERFIS/n${N}_stats.txt" 2>&1
        echo "OK: $PERFIS/n$N.nsys-rep e $PERFIS/n${N}_stats.txt"
    else
        echo "FALHA: $PERFIS/n$N.nsys-rep nao foi gerado (veja $PERFIS/n${N}_saida.txt)"
    fi
done

echo "Concluido."
echo "Perfis: $PERFIS/*.nsys-rep  |  Tabelas: $PERFIS/*_stats.txt"