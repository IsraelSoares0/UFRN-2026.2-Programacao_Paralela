# Tarefa 6: Escopo de variáveis e regiões críticas

Implemente em C a estimativa estocástica de π. Paralelize com `#pragma omp parallel for` e explique o resultado incorreto. 

Corrija a condição de corrida utilizando o `#pragma omp critical` e reestruturando com `#pragma omp parallel` seguido de `#pragma omp for` e aplicando as cláusulas private, firstprivate, lastprivate e shared. 

Teste diferentes combinações e explique como cada cláusula afeta o comportamento do programa. Comente também como a cláusula default(none) pode ajudar a tornar o escopo mais claro em programas complexos.

## Compilação e Uso

Compilação:
- `$ gcc -fopenmp -O2 -Wall -o pi_openmp pi_openmp.c -lm`

Execução:
- `./tarefa6.c`