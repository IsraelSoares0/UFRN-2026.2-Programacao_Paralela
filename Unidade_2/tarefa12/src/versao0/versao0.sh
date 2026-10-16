#!/bin/bash
#SBATCH --job-name=tarefa12_versao0
#SBATCH --time=0-0:10
#SBATCH --cpus-per-task=16
#SBATCH --partition=amd-512

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

pascalanalyzer ./versao0 -t man -g -r 3 --idtm 2 -c 1,2,4,8,16 -i 0.01,0.02,0.04,0.08,0.16,0.32 -o versao0_relatorio.json