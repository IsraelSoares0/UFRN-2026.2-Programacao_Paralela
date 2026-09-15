# Tarefa 5: Comparação entre programação sequencial e paralela

Implemente um programa em C que conte quantos números primos existem entre 2 e um valor máximo n. 
Depois, paralelize o laço principal usando a diretiva #pragma omp parallel for sem alterar a lógica original. 

Compare o tempo de execução e os resultados das versões sequencial e paralela. observe possíveis diferenças no resultado e no desempenho, e reflita sobre os desafios iniciais da programação paralela, como correção e distribuição de carga.

## Compilação e Uso

Compilação:

- `$ gcc -fopenmp tarefa5.c -o tarefa5`

Execução:

- `.\tarefa5.c`