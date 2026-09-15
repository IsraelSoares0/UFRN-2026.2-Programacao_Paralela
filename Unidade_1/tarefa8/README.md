# Tarefa 8

Implemente estimativa estocrástica de pi usando rand() para gerar os pontos. Cada thread deve usar uma variável privada para contar os acertos e acumular o total em uma variável global com

    #pragma omp critical

Depois, implemente uma segunda versão em que cada thread escreve seus acertos em uma posição distinta de um vetor compartilhado. A acumulação deve ser feita em um laço serial após a região paralela.

Compare o tempo de execução das duas versões. Em seguida, substitua rand() por rand_r() em ambas e compare novamente. Explique o comportamento dos quatro programas com base na coerência de cache e nos efeitos do falso compartilhamento.

## Compilação e Uso

Compilar (MinGW / GCC no Windows):

`gcc -O2 -fopenmp pi_monte_carlo_omp.c -o pi_monte_carlo_omp.exe`

Executar:s

`pi_monte_carlo_omp.exe [num_pontos] [num_threads]`