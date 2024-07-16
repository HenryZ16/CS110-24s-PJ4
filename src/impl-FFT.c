#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG

const double pi = 3.14159265358979323846;

inline int log_2(int n);
void print_matrix(double *p, int N);
void print_matrix_complex(double *p, int N);
void fft(double *p, int N, int type);
void fft2d(double *p, int N, int type);
void transpose(double *p, int N);
void mul_hadamard_matrix_complex(double *a, double *b, int N); // A *= B
void sum_matrix_complex(double *a, double *b, int N);          // A += B
int ceil_base_2(int n);

// Yours: 461.255421ms
void impl(int N, int step, double *p) {
  // print_matrix(p, N);

  int ceil_N = ceil_base_2(N);
  double *p_kernel = calloc(2 * ceil_N * ceil_N, sizeof(double));
  double *p_ceil_base_2 = calloc(2 * ceil_N * ceil_N, sizeof(double));
  double *p_ceil_base_2_boarder = calloc(2 * ceil_N * ceil_N, sizeof(double));
  double *p_shift = calloc(2 * ceil_N * ceil_N, sizeof(double));

#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      p_ceil_base_2[i * ceil_N * 2 + j * 2] = p[i * N + j];
    }

    p_ceil_base_2_boarder[i * 2] = p[i];
    p_ceil_base_2_boarder[i * ceil_N * 2] = p[i * N];
    p_ceil_base_2_boarder[i * ceil_N * 2 + ceil_N * 2 - 2] = p[i * N + N - 1];
    p_ceil_base_2_boarder[(ceil_N - 1) * ceil_N * 2 + i * 2] =
        p[(N - 1) * N + i];
  }

#pragma omp parallel for
  for (size_t i = 0; i < ceil_N; ++i) {
    for (size_t j = 0; j < ceil_N; ++j) {
      p_shift[i * ceil_N * 2 + j * 2] = cos(2 * pi * (i + j) / N);
      p_shift[i * ceil_N * 2 + j * 2 + 1] = -sin(2 * pi * (i + j) / N);
    }
  }

  p_kernel[1] = 0.25;
  p_kernel[ceil_N * 2] = 0.25;
  p_kernel[ceil_N * 2 + 2] = 0.25;
  p_kernel[ceil_N * 4 + 1] = 0.25;

  fft2d(p_ceil_base_2, ceil_N, 1);
  fft2d(p_ceil_base_2_boarder, ceil_N, 1);
  fft2d(p_kernel, ceil_N, 1);

  for (size_t i = 0; i < step; ++i) {
#pragma omp parallel for
    for (size_t i = 0; i < N * N * 2; i += 2) {
      double t1 = p_ceil_base_2[i] * p_kernel[i] -
                  p_ceil_base_2[i + 1] * p_kernel[i + 1]; // real
      double t2 = p_ceil_base_2[i] * p_kernel[i + 1] +
                  p_ceil_base_2[i + 1] * p_kernel[i]; // imag
      double t3 = t1 * p_shift[i] - t2 * p_shift[i + 1];
      double t4 = t1 * p_shift[i + 1] + t2 * p_shift[i];
      p_ceil_base_2[i] = t3 + p_ceil_base_2_boarder[i];
      p_ceil_base_2[i + 1] = t4 + p_ceil_base_2_boarder[i + 1];
    }
  }

  fft2d(p_ceil_base_2, ceil_N, -1);

#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      p[i * N + j] = p_ceil_base_2[i * ceil_N * 2 + j * 2];
    }
  }
  free(p_ceil_base_2);
  free(p_ceil_base_2_boarder);
  free(p_kernel);

  // print_matrix(p, N);
}

void print_matrix(double *p, int N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      printf("%f ", p[i * N + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void print_matrix_complex(double *p, int N) {
  for (size_t i = 0; i < N; ++i) {
    printf("[");
    for (size_t j = 0; j < N; ++j) {
      printf("%f + %fi,\t\t", p[i * N * 2 + j * 2], p[i * N * 2 + j * 2 + 1]);
    }
    printf("]\n");
  }
  printf("\n");
}

inline int log_2(int n) {
  int i = 0;
  while (n > 1) {
    n >>= 1;
    i++;
  }
  return i;
}

// in a iterative way
void fft(double *p, int N, int type) {
  size_t logN = log_2(N);

  // rearrangement
  for (size_t i = 0; i < N; ++i) {
    // reverse bits
    size_t j = 0;
    for (size_t k = 0; k < logN; ++k) {
      j <<= 1;
      j |= (i >> k) & 1;
    }

    if (i < j) {
      double t = p[i * 2];
      p[i * 2] = p[j * 2];
      p[j * 2] = t;
      t = p[i * 2 + 1];
      p[i * 2 + 1] = p[j * 2 + 1];
      p[j * 2 + 1] = t;
    }
  }

  // iterative part
  for (size_t mid = 1; mid < N; mid <<= 1) {
    double wn[2] = {cos(pi / mid), -type * sin(pi / mid)};

    for (size_t R = mid << 1, j = 0; j < N; j += R) {
      double w[2] = {1, 0};

      for (size_t k = 0; k < mid; ++k) {
        double x_real = p[(j + k) * 2];
        double x_imag = p[(j + k) * 2 + 1];
        double y_real =
            w[0] * p[(j + k + mid) * 2] - w[1] * p[(j + k + mid) * 2 + 1];
        double y_imag =
            w[0] * p[(j + k + mid) * 2 + 1] + w[1] * p[(j + k + mid) * 2];
        p[(j + k) * 2] = x_real + y_real;
        p[(j + k) * 2 + 1] = x_imag + y_imag;
        p[(j + k + mid) * 2] = x_real - y_real;
        p[(j + k + mid) * 2 + 1] = x_imag - y_imag;

        double t = w[0] * wn[0] - w[1] * wn[1];
        w[1] = w[0] * wn[1] + w[1] * wn[0];
        w[0] = t;
      }
    }
  }

  if (type == -1) {
    for (size_t i = 0; i < N; ++i) {
      p[i * 2] /= N;
      p[i * 2 + 1] /= N;
    }
  }
}

void fft2d(double *p, int N, int type) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    fft(p + i * N * 2, N, type);
  }
  transpose(p, N);
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    fft(p + i * N * 2, N, type);
  }
  transpose(p, N);
}

void transpose(double *p, int N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double t = p[i * N * 2 + j * 2];
      p[i * N * 2 + j * 2] = p[j * N * 2 + i * 2];
      p[j * N * 2 + i * 2] = t;
      t = p[i * N * 2 + j * 2 + 1];
      p[i * N * 2 + j * 2 + 1] = p[j * N * 2 + i * 2 + 1];
      p[j * N * 2 + i * 2 + 1] = t;
    }
  }
}

void mul_hadamard_matrix_complex(double *a, double *b, int N) {
#pragma omp parallel for
  for (size_t i = 0; i < N * N * 2; i += 2) {
    double t = a[i] * b[i] - a[i + 1] * b[i + 1]; // real
    a[i + 1] = a[i] * b[i + 1] + a[i + 1] * b[i]; // imag
    a[i] = t;
  }
}

void sum_matrix_complex(double *a, double *b, int N) {
#pragma omp parallel for
  for (size_t i = 0; i < N * N * 2; i += 2) {
    a[i] += b[i];
    a[i + 1] += b[i + 1];
  }
}

int ceil_base_2(int n) {
  int i = 1;
  while (i < n) {
    i <<= 1;
  }
  return i;
}