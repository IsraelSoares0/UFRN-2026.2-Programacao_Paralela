# Tarefa 11: Parcionamento de dados e balanceamento de carga

## Descrição

Escreva um código que simule o movimento de um fluido ao longo do tempo usando a equação de **Navier-Stokes**, considerando apenas os efeitos da viscosidade. 

- Desconsidere a pressão e quaisquer forças externas;
- Utilize diferenças finitas para discretizar o espaço e simule a evolução da velocidade do fluido no tempo;
- Inicialize o fluido parado ou com velocidade constante e verifique se o campo permanece estável.

Em seguida, crie uma pequena perturbação e observe se ela se difunde suavemente.

Após validar o código, paralelize-o com `OpenMP` e explore o impacto das cláusulas `schedule` e `collapse` no desempenho da execução paralela.

## Dados PascalSuite

### Collapse

1. Efficiency

| Cores/Entrada | i1 | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 101.510% | 92.480% | 94.410% | 95.300% |  
| 4 | 87.980% | 85.400% | 86.160% | 91.540% |
| 8 | 45.840% | 43.640% | 46.140% | 48.830% |

2. Scalability

| Cores/Entrada | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: |
| 2 | -7.520% | -5.590% | -4.700% | 
| 4 | -2.580% | -1.820% | 3.560% |
| 8 | -2.190% | 0.300% | 3.000% |

3. Speedup

| Cores/Entrada | i1 | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 2.030x | 1.850x | 1.890x | 1.910x |  
| 4 | 3.520x | 3.420x | 3.450x | 3.660x |
| 8 | 3.670x | 3.490x | 3.690x | 3.910x |

4. Performance

| Cores/Entrada | i1 | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 38.630 | 34.100 | 33.010 | 31.170 |  
| 4 | 94.460 | 96.960 | 90.910 | 91.570 |
| 8 | 100 | 100 | 100 | 100 |

5. RunTime (s)

| Cores/Entrada | i1 | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.115 | 2.107 | 4.217 | 8.379 |
| 2 | 0.549 | 1.139 | 2.233 | 4.396 |  
| 4 | 0.317 | 0.617 | 1.224 | 2.288 |
| 8 | 0.304 | 0.603 | 1.143 | 2.145 |

### Schedule Static

1. Efficiency

| Cores/Entrada | i1 | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 68.500% | 71.990% | 72.180% | 73.090% |  
| 4 | 58.470% | 58.800% | 60.050% | 59.300% |
| 8 | 34.930% | 35.500% | 35.650% | 37.100% |

2. Scalability

| Cores/Entrada | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: |
| 2 | 3.490% | 3.680% | 4.590% | 
| 4 | 0.340% | 1.580% | 0.840% |
| 8 | 0.570% | 0.720% | 2.170% |

3. Speedup

| Cores/Entrada | i1 | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 1.370x | 1.440x | 1.440x | 1.460x |  
| 4 | 2.340x | 2.350x | 2.400x | 2.370x |
| 8 | 2.790x | 2.840x | 2.850x | 2.970x |

4. Performance

| Cores/Entrada | i1 | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 20.720 | 23.900 | 23.960 | 23.460 |  
| 4 | 74.600 | 73.470 | 75.710 | 69.710 |
| 8 | 100 | 100 | 100 | 100 |

5. RunTime (s)

| Cores/Entrada | i1 | i2 | i3 | i4 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.741 | 1.486 | 3.008 | 5.878 |
| 2 | 0.541 | 1.032 | 2.084 | 4.021 |  
| 4 | 0.317 | 0.632 | 1.252 | 2.478 |
| 8 | 0.265 | 0.523 | 1.055 | 1.980 |