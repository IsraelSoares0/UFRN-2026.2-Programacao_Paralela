#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>

#define PI acos(-1.0)
#define LINE "--------------------\n"

/**
 * PARAMETROS DA COLETA DE DADOS (substituem o antigo tarefa19.sh)
 *
 * ./tarefa19            -> varre os tamanhos abaixo, REPS repeticoes cada,
 *                          e grava os tempos em ARQUIVO_CSV
 * ./tarefa19 <n> <nsteps> -> uma unica execucao (como no codigo original),
 *                          util para perfilar com nsys
 */
#ifndef REPS
#define REPS 3                       // repeticoes por tamanho
#endif
#ifndef NSTEPS
#define NSTEPS 100                   // passos de tempo na varredura
#endif
#define ARQUIVO_CSV "resultados_tarefa19.csv"

static const int TAMANHOS[] = {500, 1000, 2000, 4000, 8000};   // grade n x n
#define N_TAMANHOS ((int)(sizeof(TAMANHOS) / sizeof(TAMANHOS[0])))

/**
 * DEFINICOES DE FUNCAO
 */
void executa(int n, int nsteps, double *solve_time, double *total_time, double *norm_out);
void initial_value(const int n, const double dx, const double length, double * restrict u);
void zero(const int n, double * restrict u);
void solve(const int n, const double alpha, const double dx, const double dt, const double * restrict u, double * restrict u_tmp);
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const int n, const double * restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

/**
 * MAIN
 */
int main(int argc, char *argv[]) {

  printf("Dispositivos OpenMP: %d\n", omp_get_num_devices());

  double solve_time, total_time, norm;

  // Execucao unica, como no codigo original
  if (argc == 3) {

    // Problem size, forms an nxn grid
    int n = 1000;

    // Number of timesteps
    int nsteps = 10;

    // Set problem size from first argument
    n = atoi(argv[1]);
    if (n < 0) {
      fprintf(stderr, "Error: n must be positive\n");
      exit(EXIT_FAILURE);
    }

    // Set number of timesteps from second argument
    nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      fprintf(stderr, "Error: nsteps must be positive\n");
      exit(EXIT_FAILURE);
    }

    executa(n, nsteps, &solve_time, &total_time, &norm);
    return 0;
  }

  // Varredura de tamanhos e repeticoes
  FILE *csv = fopen(ARQUIVO_CSV, "w");
  if (!csv) {
    perror("Erro ao abrir " ARQUIVO_CSV);
    return EXIT_FAILURE;
  }
  fprintf(csv, "n,nsteps,rep,tempo_solve_s,tempo_total_s,erro_l2\n");

  // Inicializa o contexto da GPU antes das medicoes
  int dummy = 0;
  #pragma omp target map(tofrom: dummy)
  { dummy += 1; }

  for (int k = 0; k < N_TAMANHOS; ++k) {
    int n = TAMANHOS[k];

    printf("=== n=%d | aquecimento (descartado) ===\n", n);
    executa(n, NSTEPS, &solve_time, &total_time, &norm);

    for (int rep = 1; rep <= REPS; ++rep) {
      printf("=== n=%d | rep=%d ===\n", n, rep);
      executa(n, NSTEPS, &solve_time, &total_time, &norm);
      fprintf(csv, "%d,%d,%d,%lf,%lf,%E\n", n, NSTEPS, rep, solve_time, total_time, norm);
      fflush(csv);
    }
  }

  fclose(csv);
  printf("Concluido. Resultados em %s\n", ARQUIVO_CSV);
  return 0;
}


// Corpo do main original: uma execucao completa para um n e nsteps
void executa(int n, int nsteps, double *solve_time, double *total_time, double *norm_out) {

  // Start the total program runtime timer
  double start = omp_get_wtime();


  //
  // Set problem definition
  //
  double alpha = 0.1;          // heat equation coefficient
  double length = 1000.0;      // physical size of domain: length x length square
  double dx = length / (n+1);  // physical size of each cell (+1 as don't simulate boundaries as they are given)
  double dt = 0.5 / nsteps;    // time interval (total time of 0.5s)


  // Stability requires that dt/(dx^2) <= 0.5,
  double r = alpha * dt / (dx * dx);

  // Print message detailing runtime configuration
  printf("\n");
  printf(" MMS heat equation\n\n");
  printf(LINE);
  printf("Problem input\n\n");
  printf(" Grid size: %d x %d\n", n, n);
  printf(" Cell width: %E\n", dx);
  printf(" Grid length: %lf x %lf\n", length, length);
  printf("\n");
  printf(" Alpha: %E\n", alpha);
  printf("\n");
  printf(" Steps: %d\n", nsteps);
  printf(" Total time: %E\n", dt*(double)nsteps);
  printf(" Time step: %E\n", dt);
  printf(LINE);

  // Stability check
  printf("Stability\n\n");
  printf(" r value: %lf\n", r);
  if (r > 0.5)
    printf(" Warning: unstable\n");
  printf(LINE);


  // Allocate two nxn grids
  double *u     = malloc(sizeof(double)*n*n);
  double *u_tmp = malloc(sizeof(double)*n*n);
  double *tmp;

  // Set the initial value of the grid under the MMS scheme
  initial_value(n, dx, length, u);
  zero(n, u_tmp);

  //
  // Run through timesteps under the explicit scheme
  //

  // Start the solve timer
  double tic = omp_get_wtime();
  for (int t = 0; t < nsteps; ++t) {

    // Call the solve kernel
    // Computes u_tmp at the next timestep
    // given the value of u at the current timestep
    solve(n, alpha, dx, dt, u, u_tmp);

    // Pointer swap
    tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }
  // Stop solve timer
  double toc = omp_get_wtime();

  //
  // Check the L2-norm of the computed solution
  // against the *known* solution from the MMS scheme
  //
  double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

  // Stop total timer
  double stop = omp_get_wtime();

  // Print results
  printf("Results\n\n");
  printf("Error (L2norm): %E\n", norm);
  printf("Solve time (s): %lf\n", toc-tic);
  printf("Total time (s): %lf\n", stop-start);
  printf(LINE);

  // Devolve os valores para a coleta de dados
  *solve_time = toc - tic;
  *total_time = stop - start;
  *norm_out   = norm;

  // Free the memory
  free(u);
  free(u_tmp);

}


// Sets the mesh to an initial value, determined by the MMS scheme
void initial_value(const int n, const double dx, const double length, double * restrict u) {

  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx; // Physical x position
    for (int i = 0; i < n; ++i) {
      u[i+j*n] = sin(PI * x / length) * sin(PI * y / length);
      x += dx;
    }
    y += dx; // Physical y position
  }

}


// Zero the array u
void zero(const int n, double * restrict u) {

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      u[i+j*n] = 0.0;
    }
  }

}


// Compute the next timestep, given the current timestep
void solve(const int n, const double alpha, const double dx, const double dt, const double * restrict u, double * restrict u_tmp) {

  // Finite difference constant multiplier
  const double r = alpha * dt / (dx * dx);
  const double r2 = 1.0 - 4.0*r;

  // Loop over the nxn grid
  #pragma omp target teams loop collapse(2) map(tofrom: u[0:n*n], u_tmp[0:n*n])
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {

      // Update the 5-point stencil, using boundary conditions on the edges of the domain.
      // Boundaries are zero because the MMS solution is zero there.
      u_tmp[i+j*n] =  r2 * u[i+j*n] +
      r * ((i < n-1) ? u[i+1+j*n] : 0.0) +
      r * ((i > 0)   ? u[i-1+j*n] : 0.0) +
      r * ((j < n-1) ? u[i+(j+1)*n] : 0.0) +
      r * ((j > 0)   ? u[i+(j-1)*n] : 0.0);
    }
  }
}


// True answer given by the manufactured solution
double solution(const double t, const double x, const double y, const double alpha, const double length) {

  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);

}


// Computes the L2-norm of the computed grid and the MMS known solution
// The known solution is the same as the boundary function.
double l2norm(const int n, const double * restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {

  // Final (real) time simulated
  double time = dt * (double)nsteps;

  // L2-norm error
  double l2norm = 0.0;

  // Loop over the grid and compute difference of computed and known solutions as an L2-norm
  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      double answer = solution(time, x, y, alpha, length);
      l2norm += (u[i+j*n] - answer) * (u[i+j*n] - answer);

      x += dx;
    }
    y += dx;
  }

  return sqrt(l2norm);

}