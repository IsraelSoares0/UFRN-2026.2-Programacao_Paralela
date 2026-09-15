# Tarefa 3: Pipeline e Vetorização

Implemente três laços em C para investigar os efeitos do paralelismo ao nível de instrução (ILP):

1) inicialize um vetor com um cálculo simples;

2) some seus elementos de forma acumulativa, criando dependência entre as iterações;

3) quebre essa dependência utilizando múltiplas variáveis. Compare o tempo de execução das versões compiladas com diferentes níveis de otimização (O0, O2, O3) e analise como o estilo do código e as dependências influenciam o desempenho.

## Compilação e Uso

Compile com diferentes níveis de otimização e compare os tempos:

- `$ gcc -O0 -o ilp_O0 ilp_pipeline.c`
- `$ gcc -O2 -o ilp_O2 ilp_pipeline.c`
- `$ gcc -O3 -o ilp_O3 ilp_pipeline.c`

Execute várias vezes cada binário para obter uma média confiável:

- `./ilp_O0`
- `./ilp_O2`
- `./ilp_O3`
