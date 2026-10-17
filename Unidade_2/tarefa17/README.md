# Tarefa 17: Tipos Derivados em MPI

## Descrição

Reimplemente a tarefa 16, agora distribuindo as colunas entre os processos. Utilize `MPI_Type_vector` e `MPI_Type_create_resized` para 
definir um tipo derivado que represente colunas da matriz. 

Use `MPI_Scatter` com esse tipo para distribuir blocos de colunas, e `MPI_Scatter` ou cópia manual para enviar os segmentos correspondentes de x. 

Cada processo deve calcular uma contribuição parcial para todos os elementos de y e usar `MPI_Reduce` com `MPI_SUM` para somar os vetores parciais no processo 0. 

Discuta as diferenças de acesso à memória e desempenho em relação à distribuição por linhas.