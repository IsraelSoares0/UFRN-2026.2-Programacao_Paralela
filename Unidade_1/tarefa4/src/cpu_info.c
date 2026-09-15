#include <stdio.h>
#include <omp.h>

int main() {
    // Get the number of available logical processors/cores
    int cores = omp_get_num_procs();

    // Get the maximum number of threads OpenMP will use for a parallel region
    int max_threads = omp_get_max_threads();

    printf("Number of available cores: %d\n", cores);
    printf("Max OpenMP threads: %d\n", max_threads);

    #pragma omp parallel
    {
        // This only runs inside a parallel block
        #pragma omp single
        {
            printf("Running with %d active threads\n", omp_get_num_threads());
        }
    }

    return 0;
}