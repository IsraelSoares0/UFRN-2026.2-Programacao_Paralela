#!/bin/bash
# Varre threads x collapse x schedule x chunk e grava results.csv
# Uso: ./bench.sh            (valores padrão)
#      N=2048 TF=0.005 THREADS="1 4 8 16" REPS=5 ./bench.sh
set -e

N=${N:-1024}
TF=${TF:-0.01}
NU=${NU:-0.01}
REPS=${REPS:-3}
THREADS=${THREADS:-"1 2 4 8"}
SCHEDULES=${SCHEDULES:-"static dynamic guided"}
CHUNKS=${CHUNKS:-"0 1 16 256"}
EXE=${EXE:-./ns_visc_omp}

gcc -O2 -Wall -fopenmp -o ns_visc_omp ns_visc_omp.c -lm

echo "modo,N,threads,schedule,chunk,collapse,passos,t_passo,t_total,err_max,rep" > results.csv

for th in $THREADS; do
  for col in 0 1; do
    for sch in $SCHEDULES; do
      for ck in $CHUNKS; do
        for rep in $(seq 1 "$REPS"); do
          linha=$($EXE 2 "$N" "$TF" "$NU" 0 "$th" "$sch" "$ck" "$col" | grep '^RESULT')
          echo "${linha#RESULT,},$rep" >> results.csv
        done
        echo "threads=$th collapse=$col schedule=$sch chunk=$ck ... ok"
      done
    done
  done
done

echo
echo "Resultados em results.csv (use o menor t_passo entre as repeticoes)."
