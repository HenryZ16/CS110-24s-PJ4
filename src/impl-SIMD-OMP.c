#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

// Yours: 461.255421ms
// speed up: 25.007
void impl(int N, int step, double *p) {
  double *p_next = (double *)malloc(N * N * sizeof(double));
  double divisor[4] = {
      4.0f,
      4.0f,
      4.0f,
      4.0f,
  };
  __m256d p_divisor = _mm256_loadu_pd(divisor);
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    p_next[i] = p[i];
    p_next[i * N] = p[i * N];
    p_next[(N - 1) * N + i] = p[(N - 1) * N + i];
    p_next[i * N + N - 1] = p[i * N + N - 1];
  }

  for (int k = 0; k < step; k++) {
#pragma omp parallel for
    for (int i = 1; i < N - 1; i++) {
      int j = 1;
      for (; j < N - 4; j += 4) {
        _mm256_storeu_pd(
            &p_next[i * N + j],
            _mm256_div_pd(
                _mm256_add_pd(
                    _mm256_add_pd(_mm256_loadu_pd(&p[i * N + j - 1]),
                                  _mm256_loadu_pd(&p[i * N + j + 1])),
                    _mm256_add_pd(_mm256_loadu_pd(&p[(i - 1) * N + j]),
                                  _mm256_loadu_pd(&p[(i + 1) * N + j]))),
                p_divisor));
      }

      // for the tail
      for (; j < N - 1; j++) {
        p_next[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] +
                             p[i * N + j - 1] + p[i * N + j + 1]) /
                            4.0f;
      }
    }
    double *temp = p;
    p = p_next;
    p_next = temp;
  }

  if (step & 1) {
    memcpy(p, p_next, N * N * sizeof(double));
  }

  free(p_next);
}