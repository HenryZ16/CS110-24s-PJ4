#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printMatrix(int N, double *p) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", p[i * N + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// Yours: 461.255421ms
// speed up: 25.007
void impl(int N, int step, double *p) {
  // printMatrix(N, p);

  double *p_next = (double *)malloc(N * N * sizeof(double));
  double divisor16[4] = {
      16.0f,
      16.0f,
      16.0f,
      16.0f,
  };
  double divisor4[4] = {
      4.0f,
      4.0f,
      4.0f,
      4.0f,
  };
  double divisor3[4] = {
      3.0f,
      3.0f,
      3.0f,
      3.0f,
  };
  double divisor2[4] = {
      2.0f,
      2.0f,
      2.0f,
      2.0f,
  };
  __m256d p_divisor16 = _mm256_loadu_pd(divisor16);
  __m256d p_divisor4 = _mm256_loadu_pd(divisor4);
  __m256d p_divisor3 = _mm256_loadu_pd(divisor3);
  __m256d p_divisor2 = _mm256_loadu_pd(divisor2);

#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    p_next[i] = p[i];
    p_next[i * N] = p[i * N];
    p_next[(N - 1) * N + i] = p[(N - 1) * N + i];
    p_next[i * N + N - 1] = p[i * N + N - 1];
  }

  for (int k = 0; k < step; k += 2) {
    // i = 1, j = 1
    // -41-
    // 42-1
    // 1-2-
    // -1--
    double p_0_1 = 4 * p[0 * N + 1];
    double p_0_2 = p[0 * N + 2];
    double p_1_0 = 4 * p[1 * N + 0];
    double p_1_1 = 2 * p[1 * N + 1];
    double p_1_3 = p[1 * N + 3];
    double p_2_0 = p[2 * N + 0];
    double p_2_2 = 2 * p[2 * N + 2];
    double p_3_1 = p[3 * N + 1];
    p_next[1 * N + 1] =
        (p_0_1 + p_0_2 + p_1_0 + p_1_1 + p_1_3 + p_2_0 + p_2_2 + p_3_1) / 16.0f;

    // i = 1, j in [2, N - 1 - 4), SIMD 256d
    // -141-
    // 1-3-1
    // -2-2-
    // --1--
    int j = 2;
    for (; j < N - 1 - 4; j += 4) {
      __m256d p_0_j_minus_1 = _mm256_loadu_pd(&p[0 * N + j - 1]);
      __m256d p_0_j = _mm256_mul_pd(_mm256_loadu_pd(&p[0 * N + j]), p_divisor4);
      __m256d p_0_j_plus_1 = _mm256_loadu_pd(&p[0 * N + j + 1]);
      __m256d p_1_j_minus_2 = _mm256_loadu_pd(&p[1 * N + j - 2]);
      __m256d p_1_j = _mm256_mul_pd(_mm256_loadu_pd(&p[1 * N + j]), p_divisor3);
      __m256d p_1_j_plus_2 = _mm256_loadu_pd(&p[1 * N + j + 2]);
      __m256d p_2_j_minus_1 =
          _mm256_mul_pd(_mm256_loadu_pd(&p[2 * N + j - 1]), p_divisor2);
      __m256d p_2_j_plus_1 =
          _mm256_mul_pd(_mm256_loadu_pd(&p[2 * N + j + 1]), p_divisor2);
      __m256d p_3_j = _mm256_loadu_pd(&p[3 * N + j]);

      __m256d sum_1 = _mm256_add_pd(_mm256_add_pd(p_0_j_minus_1, p_0_j),
                                    _mm256_add_pd(p_0_j_plus_1, p_1_j_minus_2));
      __m256d sum_2 = _mm256_add_pd(_mm256_add_pd(p_1_j, p_1_j_plus_2),
                                    _mm256_add_pd(p_2_j_minus_1, p_2_j_plus_1));
      __m256d sum_3 = _mm256_add_pd(sum_1, sum_2);
      __m256d sum = _mm256_add_pd(sum_3, p_3_j);
      _mm256_storeu_pd(&p_next[1 * N + j], _mm256_div_pd(sum, p_divisor16));
    }
    for (; j < N - 2; ++j) {
      double p_0_j_minus_1 = p[0 * N + j - 1];
      double p_0_j = 4 * p[0 * N + j];
      double p_0_j_plus_1 = p[0 * N + j + 1];
      double p_1_j_minus_2 = p[1 * N + j - 2];
      double p_1_j = 3 * p[1 * N + j];
      double p_1_j_plus_2 = p[1 * N + j + 2];
      double p_2_j_minus_1 = 2 * p[2 * N + j - 1];
      double p_2_j_plus_1 = 2 * p[2 * N + j + 1];
      double p_3_j = p[3 * N + j];
      p_next[1 * N + j] =
          (p_0_j_minus_1 + p_0_j + p_0_j_plus_1 + p_1_j_minus_2 + p_1_j +
           p_1_j_plus_2 + p_2_j_minus_1 + p_2_j_plus_1 + p_3_j) /
          16.0f;
    }

    // i = 1, j = N - 2
    // 0 -14-
    // 1 1-24
    // 2 -2-1
    // 3 --1-
    double p_0_N_minus_3 = p[0 * N + N - 3];
    double p_0_N_minus_2 = 4 * p[0 * N + N - 2];
    double p_1_N_minus_4 = p[1 * N + N - 4];
    double p_1_N_minus_2 = 2 * p[1 * N + N - 2];
    double p_1_N_minus_1 = 4 * p[1 * N + N - 1];
    double p_2_N_minus_3 = 2 * p[2 * N + N - 3];
    double p_2_N_minus_1 = p[2 * N + N - 1];
    double p_3_N_minus_2 = p[3 * N + N - 2];
    p_next[1 * N + N - 2] =
        (p_0_N_minus_3 + p_0_N_minus_2 + p_1_N_minus_4 + p_1_N_minus_2 +
         p_1_N_minus_1 + p_2_N_minus_3 + p_2_N_minus_1 + p_3_N_minus_2) /
        16.0f;

    // i in [2, N - 2)
#pragma omp parallel for
    for (int i = 2; i < N - 2; i++) {
      // i = i, j = 1
      // i-2 -1--
      // i-1 1-2-
      // i+0 43-1
      // i+1 1-2-
      // i+2 -1--
      double p_i_minus_2_1 = p[(i - 2) * N + 1];
      double p_i_minus_1_0 = p[(i - 1) * N + 0];
      double p_i_minus_1_2 = 2 * p[(i - 1) * N + 2];
      double p_i_0 = 4 * p[i * N + 0];
      double p_i_1 = 3 * p[i * N + 1];
      double p_i_3 = p[i * N + 3];
      double p_i_plus_1_0 = p[(i + 1) * N + 0];
      double p_i_plus_1_2 = 2 * p[(i + 1) * N + 2];
      double p_i_plus_2_1 = p[(i + 2) * N + 1];
      p_next[i * N + 1] =
          (p_i_minus_2_1 + p_i_minus_1_0 + p_i_minus_1_2 + p_i_0 + p_i_1 +
           p_i_3 + p_i_plus_1_0 + p_i_plus_1_2 + p_i_plus_2_1) /
          16.0f;

      // i = i, j in [2, N - 1 - 4), SIMD 256d
      // --1--
      // -2-2-
      // 1-4-1
      // -2-2-
      // --1--
      int j = 2;
      for (; j < N - 1 - 4; j += 4) {
        __m256d p_i_minus_2_j = _mm256_loadu_pd(&p[(i - 2) * N + j]);
        __m256d p_i_minus_1_j_minus_1 =
            _mm256_mul_pd(_mm256_loadu_pd(&p[(i - 1) * N + j - 1]), p_divisor2);
        __m256d p_i_minus_1_j_plus_1 =
            _mm256_mul_pd(_mm256_loadu_pd(&p[(i - 1) * N + j + 1]), p_divisor2);
        __m256d p_i_0_j_minus_2 = _mm256_loadu_pd(&p[i * N + j - 2]);
        __m256d p_i_0_j =
            _mm256_mul_pd(_mm256_loadu_pd(&p[i * N + j]), p_divisor4);
        __m256d p_i_0_j_plus_2 = _mm256_loadu_pd(&p[i * N + j + 2]);
        __m256d p_i_plus_1_j_minus_1 =
            _mm256_mul_pd(_mm256_loadu_pd(&p[(i + 1) * N + j - 1]), p_divisor2);
        __m256d p_i_plus_1_j_plus_1 =
            _mm256_mul_pd(_mm256_loadu_pd(&p[(i + 1) * N + j + 1]), p_divisor2);
        __m256d p_i_plus_2_j = _mm256_loadu_pd(&p[(i + 2) * N + j]);

        __m256d sum_1 =
            _mm256_add_pd(p_i_minus_2_j, _mm256_add_pd(p_i_minus_1_j_minus_1,
                                                       p_i_minus_1_j_plus_1));
        __m256d sum_2 =
            _mm256_add_pd(_mm256_add_pd(p_i_0_j_minus_2, p_i_0_j),
                          _mm256_add_pd(p_i_0_j_plus_2, p_i_plus_1_j_minus_1));
        __m256d sum_3 = _mm256_add_pd(
            _mm256_add_pd(p_i_plus_1_j_plus_1, p_i_plus_2_j), sum_1);
        __m256d sum = _mm256_add_pd(sum_2, sum_3);
        _mm256_storeu_pd(&p_next[i * N + j], _mm256_div_pd(sum, p_divisor16));
      }
      for (; j < N - 2; ++j) {
        double p_i_minus_2_j = p[(i - 2) * N + j];
        double p_i_minus_1_j_minus_1 = 2 * p[(i - 1) * N + j - 1];
        double p_i_minus_1_j_plus_1 = 2 * p[(i - 1) * N + j + 1];
        double p_i_0_j_minus_2 = p[i * N + j - 2];
        double p_i_0_j = 4 * p[i * N + j];
        double p_i_0_j_plus_2 = p[i * N + j + 2];
        double p_i_plus_1_j_minus_1 = 2 * p[(i + 1) * N + j - 1];
        double p_i_plus_1_j_plus_1 = 2 * p[(i + 1) * N + j + 1];
        double p_i_plus_2_j = p[(i + 2) * N + j];
        p_next[i * N + j] =
            (p_i_minus_2_j + p_i_minus_1_j_minus_1 + p_i_minus_1_j_plus_1 +
             p_i_0_j_minus_2 + p_i_0_j + p_i_0_j_plus_2 + p_i_plus_1_j_minus_1 +
             p_i_plus_1_j_plus_1 + p_i_plus_2_j) /
            16.0f;
      }

      // i = i, j = N - 2
      // --1-
      // -2-1
      // 1-34
      // -2-1
      // --1-
      double p_i_minus_2_N_minus_2 = p[(i - 2) * N + N - 1];
      double p_i_minus_1_N_minus_3 = 2 * p[(i - 1) * N + N - 3];
      double p_i_minus_1_N_minus_1 = p[(i - 1) * N + N - 1];
      double p_i_N_minus_4 = p[i * N + N - 4];
      double p_i_N_minus_2 = 3 * p[i * N + N - 2];
      double p_i_N_minus_1 = 4 * p[i * N + N - 1];
      double p_i_plus_1_N_minus_3 = 2 * p[(i + 1) * N + N - 3];
      double p_i_plus_1_N_minus_1 = p[(i + 1) * N + N - 1];
      double p_i_plus_2_N_minus_2 = p[(i + 2) * N + N - 2];
      p_next[i * N + N - 2] =
          (p_i_minus_2_N_minus_2 + p_i_minus_1_N_minus_3 +
           p_i_minus_1_N_minus_1 + p_i_N_minus_4 + p_i_N_minus_2 +
           p_i_N_minus_1 + p_i_plus_1_N_minus_3 + p_i_plus_1_N_minus_1 +
           p_i_plus_2_N_minus_2) /
          16.0f;
    }

    // i = N - 2, j = 1
    // N-4 -1--
    // N-3 1-2-
    // N-2 42-1
    // N-1 -41-
    double p_N_minus_4_1 = p[(N - 4) * N + 1];
    double p_N_minus_3_0 = p[(N - 3) * N + 0];
    double p_N_minus_3_2 = 2 * p[(N - 3) * N + 2];
    double p_N_minus_2_0 = 4 * p[(N - 2) * N + 0];
    double p_N_minus_2_1 = 2 * p[(N - 2) * N + 1];
    double p_N_minus_2_3 = p[(N - 2) * N + 3];
    double p_N_minus_1_1 = 4 * p[(N - 1) * N + 1];
    double p_N_minus_1_2 = p[(N - 1) * N + 2];
    p_next[(N - 2) * N + 1] =
        (p_N_minus_4_1 + p_N_minus_3_0 + p_N_minus_3_2 + p_N_minus_2_0 +
         p_N_minus_2_1 + p_N_minus_2_3 + p_N_minus_1_1 + p_N_minus_1_2) /
        16.0f;

    // i = N - 2, j in [2, N - 1 - 4), SIMD 256d
    // N-4 --1--
    // N-3 -2-2-
    // N-2 1-3-1
    // N-1 -141-
    j = 2;
    for (; j < N - 1 - 4; j += 4) {
      __m256d p_N_minus_4_j = _mm256_loadu_pd(&p[(N - 4) * N + j]);
      __m256d p_N_minus_3_j_minus_1 =
          _mm256_mul_pd(_mm256_loadu_pd(&p[(N - 3) * N + j - 1]), p_divisor2);
      __m256d p_N_minus_3_j_plus_1 =
          _mm256_mul_pd(_mm256_loadu_pd(&p[(N - 3) * N + j + 1]), p_divisor2);
      __m256d p_N_minus_2_j_minus_2 = _mm256_loadu_pd(&p[(N - 2) * N + j - 2]);
      __m256d p_N_minus_2_j =
          _mm256_mul_pd(_mm256_loadu_pd(&p[(N - 2) * N + j]), p_divisor3);
      __m256d p_N_minus_2_j_plus_2 = _mm256_loadu_pd(&p[(N - 2) * N + j + 2]);
      __m256d p_N_minus_1_j_minus_1 = _mm256_loadu_pd(&p[(N - 1) * N + j - 1]);
      __m256d p_N_minus_1_j =
          _mm256_mul_pd(_mm256_loadu_pd(&p[(N - 1) * N + j]), p_divisor4);
      __m256d p_N_minus_1_j_plus_1 = _mm256_loadu_pd(&p[(N - 1) * N + j + 1]);

      __m256d sum_1 =
          _mm256_add_pd(p_N_minus_4_j, _mm256_add_pd(p_N_minus_3_j_minus_1,
                                                     p_N_minus_3_j_plus_1));
      __m256d sum_2 = _mm256_add_pd(
          _mm256_add_pd(p_N_minus_2_j_minus_2, p_N_minus_2_j),
          _mm256_add_pd(p_N_minus_2_j_plus_2, p_N_minus_1_j_minus_1));
      __m256d sum_3 = _mm256_add_pd(
          _mm256_add_pd(p_N_minus_1_j, p_N_minus_1_j_plus_1), sum_1);
      __m256d sum = _mm256_add_pd(sum_2, sum_3);
      _mm256_storeu_pd(&p_next[(N - 2) * N + j],
                       _mm256_div_pd(sum, p_divisor16));
    }
    for (; j < N - 2; ++j) {
      double p_N_minus_4_j = p[(N - 4) * N + j];
      double p_N_minus_3_j_minus_1 = 2 * p[(N - 3) * N + j - 1];
      double p_N_minus_3_j_plus_1 = 2 * p[(N - 3) * N + j + 1];
      double p_N_minus_2_j_minus_2 = p[(N - 2) * N + j - 2];
      double p_N_minus_2_j = 3 * p[(N - 2) * N + j];
      double p_N_minus_2_j_plus_2 = p[(N - 2) * N + j + 2];
      double p_N_minus_1_j_minus_1 = p[(N - 1) * N + j - 1];
      double p_N_minus_1_j = 4 * p[(N - 1) * N + j];
      double p_N_minus_1_j_plus_1 = p[(N - 1) * N + j + 1];
      p_next[(N - 2) * N + j] =
          (p_N_minus_4_j + p_N_minus_3_j_minus_1 + p_N_minus_3_j_plus_1 +
           p_N_minus_2_j_minus_2 + p_N_minus_2_j + p_N_minus_2_j_plus_2 +
           p_N_minus_1_j_minus_1 + p_N_minus_1_j + p_N_minus_1_j_plus_1) /
          16.0f;
    }

    // i = N - 2. j = N - 2
    // N-4 --1-
    // N-3 -2-1
    // N-2 1-24
    // N-1 -14-
    double p_N_minus_4_N_minus_2 = p[(N - 4) * N + N - 2];
    double p_N_minus_3_N_minus_3 = 2 * p[(N - 3) * N + N - 3];
    double p_N_minus_3_N_minus_1 = p[(N - 3) * N + N - 1];
    double p_N_minus_2_N_minus_4 = p[(N - 2) * N + N - 4];
    double p_N_minus_2_N_minus_2 = 2 * p[(N - 2) * N + N - 2];
    double p_N_minus_2_N_minus_1 = 4 * p[(N - 2) * N + N - 1];
    double p_N_minus_1_N_minus_3 = p[(N - 1) * N + N - 3];
    double p_N_minus_1_N_minus_2 = 4 * p[(N - 1) * N + N - 2];
    p_next[(N - 2) * N + N - 2] =
        (p_N_minus_4_N_minus_2 + p_N_minus_3_N_minus_3 + p_N_minus_3_N_minus_1 +
         p_N_minus_2_N_minus_4 + p_N_minus_2_N_minus_2 + p_N_minus_2_N_minus_1 +
         p_N_minus_1_N_minus_3 + p_N_minus_1_N_minus_2) /
        16.0f;

    double *temp = p;
    p = p_next;
    p_next = temp;
  }

  // printMatrix(N, p);
  // printMatrix(N, p_next);

  if (step % 4 == 0 || step % 4 == 2) {
    memcpy(p_next, p, N * N * sizeof(double));
  } else {
    memcpy(p, p_next, N * N * sizeof(double));
  }

  if (step % 4 == 1 || step % 4 == 2) {
    double *temp = p;
    p = p_next;
    p_next = temp;
  }

  // printMatrix(N, p);

  free(p_next);
}