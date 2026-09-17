# Tarefa 14

## Descrição

Implemente um programa MPI com exatamente **dois processos**. O processo 0 deve enviar uma mensagem ao processo 1, que imediatamente responde com a mesma mensagem.

Meça o tempo total de execução de múltiplas trocas consecutivas dessa mensagem, utilizando `MPI_Wtime`.

Registre os tempos para diferentes tamanhos, desde mensagens pequenas (*como 8 bytes*) até mensagens maiores (*como 1 MB ou mais*).

Analise **graficamente** o tempo em função do tamanho da mensagem e identifique os regimes onde a latência domina e onde a largura de banda se torna o fator principal.

OLHAR: NPAD UFRN