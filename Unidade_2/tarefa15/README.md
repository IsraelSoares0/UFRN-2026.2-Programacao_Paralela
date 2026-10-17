# Tarefa 15: Sobreposição de computação e comunicação

## Descrição

Implemente uma simulação de difusão de calor em uma barra `1D`, dividia entre dois ou mais processos MPI.

Cada processo deve simular um trecho da barra com células extras para troca de boardas com vizinhos.

Implemente três versões:

1. V1: `MPI_Send / MPI_Recv`;
2. V2: `MPI_Isend / MPI_Irecv e MPI_Wait`;
3. V3: `MPI_Test` para atualizar os pontos internos enquanto aguarda a comunicação.

Compare os tempos de execução e discuta os ganhos com sobreposição de comunicação e computação.

--> Prestar atenção com células fantasmas (Ghost cells) -- Necessário enviar e receber células do halo com Send e Recv.

## Informações para o relatório

- Comentar sobre os impactos são versões nos tempos de execução.
- Comentar sobre como a V1, por ser bloquante, aumenta o tempo de execução conforme aumentamos a quantidade de processos, mesmo para o mesmo N.
- Falar sobre os impactos das versões 2 e 3.
- Discutir sobre os ganhos com sobreposição de comunicação e computação