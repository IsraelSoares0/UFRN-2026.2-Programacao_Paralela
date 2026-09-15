# Tarefa 2: Localidade temporal e espacial

Implemente duas versões da multiplicação de matriz por vetor (MxV) em C:

- Versão 1 (por linhas):  laço externo percorre linhas i,
                        laço interno percorre colunas j
                        -> acessa `A[i][j]` sequencialmente (stride 1)

- Versão 2 (por colunas): laço externo percorre colunas j,
                        laço interno percorre linhas i
                        -> acessa `A[i][j]` pulando N elementos a cada armazenada em ordem row-major.

Meça o tempo de execução de cada versão usando uma função apropriada e execute testes com matrizes de diferentes tamanhos.

Identifique a partir de que tamanho os tempos passam a divergir significativamente e explique por que isso ocorre, relacionando suas observações ao uso da memória cache e ao padrão de acesso à memória.

Compilação: 

- `gcc -O2 -o mxv mxv.c`

Execução: 

- `./mxv`