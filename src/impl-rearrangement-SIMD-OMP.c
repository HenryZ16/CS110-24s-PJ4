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

// Baseline: 7751.182514ms
// Yours: 197.916104ms
// speed up: 39.164
void impl(int N, int step, double *p) {
  double divisor[4] = {
      4.0f,
      4.0f,
      4.0f,
      4.0f,
  };
  __m256d p_divisor = _mm256_loadu_pd(divisor);

  // rearrange
  int N2 = (N + 1) / 2;
  double *p_part[2] = {
      malloc(N2 * N * sizeof(double)),
      malloc(N2 * N * sizeof(double)),
  };
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      p_part[(i + j) & 1][i * N2 + j / 2] = p[i * N + j];
    }
  }

  // caculate
  int INPUTpartID = 1;
  int OUTPUTpartID = 0;

  if (N & 1) { // N = odd
    for (int k = 0; k < step; k++) {
#pragma omp parallel for
      for (int i = 1; i < N - 1; i++) {
        int j_begin = (INPUTpartID + i) & 1;
        int j_end = N2 - 1;
        // for (int j = j_begin; j < j_end; ++j) {
        //   double p1 = p_part[INPUTpartID][(i - 1) * N2 + j];
        //   double p2 = p_part[INPUTpartID][i * N2 + j - j_begin];
        //   double p3 = p_part[INPUTpartID][i * N2 + 1 + j - j_begin];
        //   double p4 = p_part[INPUTpartID][(i + 1) * N2 + j];
        //   p_part[OUTPUTpartID][i * N2 + j] = (p1 + p2 + p3 + p4) / 4.0f;
        // }
        int j = j_begin;
        for (; j < j_end - 3; j += 4) {
          __m256d p1 = _mm256_loadu_pd(&p_part[INPUTpartID][(i - 1) * N2 + j]);
          __m256d p2 =
              _mm256_loadu_pd(&p_part[INPUTpartID][i * N2 + j - j_begin]);
          __m256d p3 =
              _mm256_loadu_pd(&p_part[INPUTpartID][i * N2 + 1 + j - j_begin]);
          __m256d p4 = _mm256_loadu_pd(&p_part[INPUTpartID][(i + 1) * N2 + j]);
          __m256d sum1 = _mm256_add_pd(p1, p2);
          __m256d sum2 = _mm256_add_pd(p3, p4);
          __m256d sum3 = _mm256_add_pd(sum1, sum2);
          __m256d result = _mm256_div_pd(sum3, p_divisor);
          _mm256_storeu_pd(&p_part[OUTPUTpartID][i * N2 + j], result);
        }

        // for the tail
        for (; j < j_end; j++) {
          double p1 = p_part[INPUTpartID][(i - 1) * N2 + j];
          double p2 = p_part[INPUTpartID][i * N2 + j - j_begin];
          double p3 = p_part[INPUTpartID][i * N2 + 1 + j - j_begin];
          double p4 = p_part[INPUTpartID][(i + 1) * N2 + j];
          p_part[OUTPUTpartID][i * N2 + j] = (p1 + p2 + p3 + p4) / 4.0f;
        }
      }

      int temp = INPUTpartID;
      INPUTpartID = OUTPUTpartID;
      OUTPUTpartID = temp;
    }
  } else { // N = even
    for (int k = 0; k < step; k++) {
#pragma omp parallel for
      for (int i = 1; i < N - 1; i++) {
        int j_begin = (INPUTpartID + i) & 1;
        int j_end = N2 - 1 + j_begin;
        // for (int j = j_begin; j < j_end; ++j) {
        //   double p1 = p_part[INPUTpartID][(i - 1) * N2 + j];
        //   double p2 = p_part[INPUTpartID][i * N2 + j - j_begin];
        //   double p3 = p_part[INPUTpartID][i * N2 + 1 + j - j_begin];
        //   double p4 = p_part[INPUTpartID][(i + 1) * N2 + j];
        //   p_part[OUTPUTpartID][i * N2 + j] = (p1 + p2 + p3 + p4) / 4.0f;
        // }
        int j = j_begin;
        for (; j < j_end - 3; j += 4) {
          __m256d p1 = _mm256_loadu_pd(&p_part[INPUTpartID][(i - 1) * N2 + j]);
          __m256d p2 =
              _mm256_loadu_pd(&p_part[INPUTpartID][i * N2 + j - j_begin]);
          __m256d p3 =
              _mm256_loadu_pd(&p_part[INPUTpartID][i * N2 + 1 + j - j_begin]);
          __m256d p4 = _mm256_loadu_pd(&p_part[INPUTpartID][(i + 1) * N2 + j]);
          __m256d sum1 = _mm256_add_pd(p1, p2);
          __m256d sum2 = _mm256_add_pd(p3, p4);
          __m256d sum3 = _mm256_add_pd(sum1, sum2);
          __m256d result = _mm256_div_pd(sum3, p_divisor);
          _mm256_storeu_pd(&p_part[OUTPUTpartID][i * N2 + j], result);
        }

        // for the tail
        for (; j < j_end; j++) {
          double p1 = p_part[INPUTpartID][(i - 1) * N2 + j];
          double p2 = p_part[INPUTpartID][i * N2 + j - j_begin];
          double p3 = p_part[INPUTpartID][i * N2 + 1 + j - j_begin];
          double p4 = p_part[INPUTpartID][(i + 1) * N2 + j];
          p_part[OUTPUTpartID][i * N2 + j] = (p1 + p2 + p3 + p4) / 4.0f;
        }
      }

      int temp = INPUTpartID;
      INPUTpartID = OUTPUTpartID;
      OUTPUTpartID = temp;
    }
  }

  // rearrange back
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      p[i * N + j] = p_part[(i + j) & 1][i * N2 + j / 2];
    }
  }

  // printMatrix(N, p);

  free(p_part[0]);
  free(p_part[1]);
}