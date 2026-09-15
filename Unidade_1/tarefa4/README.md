# Tarefa 4: Aplicações limitadas por memória ou CPU

Implemente dois programas paralelos em C com OpenMP:

- Programa 1: Limitado por memória, com somas simples em vetores
- Programa 2: Limitado por CPU, com cálculos matemáticos intensivos

Paralelize com:

`#pragma omp parallel for`

Meça o tempo de execução variando o número de threads. Analise quando o desempenho melhora, estabiliza ou piora, e reflita sobre como o multithreading de hardware pode ajudar em programas memory-bound, mas atrapalhar em programas compute-bound pela competição por recursos.
