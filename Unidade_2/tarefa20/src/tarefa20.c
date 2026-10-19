#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>

// Key constants used in this program
#define PI acos(-1.0) // Pi
#define LINE "--------------------\n" // A line for fancy output

// Arquivo CSV de saída com os resultados de todas as variações
#define ARQUIVO_CSV "resultados_tarefa20.csv"

// Function definitions
void initial_value(const int n, const double dx, const double length, double * restrict u);
void zero(const int n, double * restrict u);
void solve(const int n, const double alpha, const double dx, const double dt, const double * restrict u, double * restrict u_tmp);
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const int n, const double * restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);
void executa(const int n, const int nsteps, double *solve_time, double *total_time, double *norm);

// Main function
// Sem argumentos: roda todas as variações definidas abaixo e salva no CSV.
// Com 2 argumentos (n nsteps): roda apenas essa configuração (como no original).
int main(int argc, char *argv[]) {

  // ---------- Variações a executar ----------
  int tamanhos[] = {500, 1000, 2000, 4000, 8000}; // grade n x n
  int ntam = 5;
  int passos[] = {100};                           // passos de tempo
  int npassos = 1;
  int reps = 3;                                   // repetições por configuração

  // Check for the correct number of arguments
  if (argc == 3) {

    // Set problem size from first argument
    int n = atoi(argv[1]);
    if (n < 0) {
      fprintf(stderr, "Error: n must be positive\n");
      exit(EXIT_FAILURE);
    }

    // Set number of timesteps from second argument
    int nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      fprintf(stderr, "Error: nsteps must be positive\n");
      exit(EXIT_FAILURE);
    }

    tamanhos[0] = n;
    ntam = 1;
    passos[0] = nsteps;
    npassos = 1;
  }
  int acrescentar = 0;
  if (argc == 3) {
    FILE *existente = fopen(ARQUIVO_CSV, "r");
    if (existente != NULL) {
      acrescentar = 1;
      fclose(existente);
    }
  }

  FILE *csv = fopen(ARQUIVO_CSV, acrescentar ? "a" : "w");
  if (csv == NULL) {
    fprintf(stderr, "Error: nao foi possivel criar %s\n", ARQUIVO_CSV);
    exit(EXIT_FAILURE);
  }
  if (!acrescentar)
    fprintf(csv, "n,nsteps,rep,tempo_solve_s,tempo_total_s,erro_l2\n");

  // Aquecimento
  int aquece = 0;
  #pragma omp target map(tofrom: aquece)
  {
    aquece += 1;
  }

  for (int a = 0; a < ntam; ++a) {
    for (int b = 0; b < npassos; ++b) {
      for (int rep = 1; rep <= reps; ++rep) {

        printf("=== n=%d | nsteps=%d | rep=%d ===\n", tamanhos[a], passos[b], rep);

        double solve_time, total_time, norm;
        executa(tamanhos[a], passos[b], &solve_time, &total_time, &norm);

        fprintf(csv, "%d,%d,%d,%lf,%lf,%E\n",
                tamanhos[a], passos[b], rep, solve_time, total_time, norm);
        fflush(csv);
      }
    }
  }

  fclose(csv);
  printf("Resultados salvos em %s\n", ARQUIVO_CSV);

  return 0;
}


// Uma execução completa do problema (corpo do main original)
void executa(const int n, const int nsteps, double *solve_time, double *total_time, double *norm_out) {

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

  // Start the solve timer
  double tic = omp_get_wtime();
  #pragma omp target enter data map(to: u[0:n*n], u_tmp[0:n*n])
  for (int t = 0; t < nsteps; ++t) {
    solve(n, alpha, dx, dt, u, u_tmp);

    // Pointer swap
    tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }
  // u_tmp também é liberado na GPU: como agora há várias execuções no mesmo
  // processo, o mapeamento antigo não pode ficar para trás.
  #pragma omp target exit data map(from: u[0:n*n]) map(release: u_tmp[0:n*n])
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

  *solve_time = toc - tic;
  *total_time = stop - start;
  *norm_out = norm;

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