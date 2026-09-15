# Tarefa 7: Processamento Paralelo de Lista Encadeada com OpenMP Task

Implemente um programa em C que cria uma lista encadeada com nós, cada um, contendo o nome de um arquivo fictício. 

Dentro de uma região paralela, percorra a lista e crie uma tarefa com `#pragma omp task` para processar cada nó. 
Cada tarefa deve imprimir o nome do arquivo e o identificador da thread que a executou com o `omp_get_thread_num`. 

Após executar o programa, reflita
- Todos os nós foram processados? 
- Algum foi processado mais de uma vez ou ignorado? 
- O comportamento muda entre execuções? 
- Como garantir que cada nó seja processado uma única vez e por apenas uma tarefa?

## Compilação e Uso

Compilar:
- `gcc -fopenmp -Wall -o tarefa7 tarefa7.c`

Execução:
- `./tarefa7.exe` 