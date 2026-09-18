# Tarefa 16

## Descrição

Implemente um programa MPI que calcule o produto `y=A*x`, onde A é uma matriz `MxN` e x é um vetor de tamanho `N`.

Divida a matriz A por linhas entre os processos com `MPI_Scatter`, e distribua o vetor x inteiro com `MPI_Bcast`. Cada processo deve calcular os elementos de y correspondentes às suas linhas e enviá-los de volta ao processo 0 com `MPI_Gather`.

Compare os tempos com diferentes tamanhos de matriz e número de processos.