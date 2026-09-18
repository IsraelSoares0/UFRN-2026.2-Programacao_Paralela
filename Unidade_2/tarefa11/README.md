# Tarefa 11: Parcionamento de dados e balanceamento de carga

## Descrição

Escreva um código que simule o movimento de um fluido ao longo do tempo usando a equação de **Navier-Stokes**, considerando apenas os efeitos da viscosidade. 

- Desconsidere a pressão e quaisquer forças externas;
- Utilize diferenças finitas para discretizar o espaço e simule a evolução da velocidade do fluido no tempo;
- Inicialize o fluido parado ou com velocidade constante e verifique se o campo permanece estável.

Em seguida, crie uma pequena perturbação e observe se ela se difunde suavemente.

Após validar o código, paralelize-o com `OpenMP` e explore o impacto das cláusulas `schedule` e `collapse` no desempenho da execução paralela.