#include <omp.h>
#include <stdlib.h>

// Baseline: 11,534.560975ms
// Yours: 771.940519ms with 14 threads
// speed up: 14.942
void impl(int N, int step, double *p) {
  double *p_next = (double *)malloc(N * N * sizeof(double));
  for (int i = 0; i < N; ++i) {
    p_next[i] = p[i];
    p_next[i * N] = p[i * N];
    p_next[(N - 1) * N + i] = p[(N - 1) * N + i];
    p_next[i * N + N - 1] = p[i * N + N - 1];
  }
  for (int k = 0; k < step; k++) {
#pragma omp parallel for
    for (int i = 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++) {
        p_next[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] +
                             p[i * N + j - 1] + p[i * N + j + 1]) /
                            4.0f;
      }
    }
    double *temp = p;
    p = p_next;
    p_next = temp;
  }
}