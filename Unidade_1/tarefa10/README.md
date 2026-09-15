# Tarefa 10: Mecanismos de Sincronização com OpenMP

## Descrição

Implemente novamente o estimador da tarefa 8 usando um contador compartilhado e o rand_r protegendo a soma de hits com #pragma omp critical e com #pragma omp atomic. 

Compare essas duas implementações com suas versões que usam contadores privados. Agora, compare essas com uma 5ª versão que utiliza apenas a cláusula reduction ao invés das diretivas de sincronização.  

Reflita sobre a aplicabilidade de desses mecanismos em termos de desempenho e produtividade e proponha um roteiro para quando utilizar qual mecanismo de sincronização, incluindo critical nomeadas e locks explícitos.