// Copyright 2017, Brown University, Providence, RI.
//
//                         All Rights Reserved
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose other than its incorporation into a
// commercial product or service is hereby granted without fee, provided
// that the above copyright notice appear in all copies and that both
// that copyright notice and this permission notice appear in supporting
// documentation, and that the name of Brown University not be used in
// advertising or publicity pertaining to distribution of the software
// without specific, written prior permission.
//
// BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
// INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
// PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
// ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
//
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.1
//
// This file is part of MGARD.
//
// MGARD is distributed under the OSI-approved Apache License, Version 2.0.
// See accompanying file Copyright.txt for details.
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgard_api.h"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

void print_usage_message(char *argv[], FILE *fp) {
  fprintf(fp, "Usage: %s infile outfile nrow ncol nfib tolerance s\n", argv[0]);
}

void print_for_more_details_message(char *argv[], FILE *fp) {
  fprintf(fp, "\nFor more details, run: %s --help\n", argv[0]);
}

void print_help_message(char *argv[], FILE *fp) {
  fprintf(fp, "\nThe input file `infile` should contain a "
              "double[`nrow`][`ncol`][`nfib`] array.\n"
              "The array will be compressed so that the error as measured in "
              "the H^`s` norm is\n"
              "no more than `tolerance`. (Use `s` = inf for the L^infty norm "
              "and `s` = 0 for the\n"
              "L^2 norm.) The compressed array will be written to the output "
              "file `outfile`.\n");
}

int main(int argc, char *argv[]) {
  size_t result;

  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    print_usage_message(argv, stdout);
    print_help_message(argv, stdout);
    return 0;
  }

  char *infile, *outfile;
  int nrow, ncol, nfib;
  double tol, s;
  if (argc != 8) {
    if (argc < 8) {
      fprintf(stderr, "%s: Not enough arguments! ", argv[0]);
    } else {
      fprintf(stderr, "%s: Too many arguments! ", argv[0]);
    }
    print_usage_message(argv, stderr);
    print_for_more_details_message(argv, stderr);
    return 1;
  } else {
    infile = argv[1];
    outfile = argv[2];
    nrow = atoi(argv[3]);
    ncol = atoi(argv[4]);
    nfib = atoi(argv[5]);
    tol = atof(argv[6]);
    s = atof(argv[7]);
  }

  long lSize;
  double *buffer;
  long num_doubles;

  if (0) {
    num_doubles = nrow * ncol * nfib;
    long num_bytes = sizeof(double) * num_doubles;
    lSize = num_bytes;

    buffer = (double *)malloc(sizeof(char) * lSize);
    if (buffer == NULL) {
      fputs("Memory error", stderr);
      exit(2);
    }

    for (int i = 0; i < num_doubles; i++) {
      buffer[i] = rand() % 10 + 1;
    }

  } else {

    num_doubles = nrow * ncol * nfib;
    long num_bytes = sizeof(double) * num_doubles;

    FILE *pFile;
    pFile = fopen(infile, "rb");
    if (pFile == NULL) {
      fputs("File error", stderr);
      exit(1);
    }
    fseek(pFile, 0, SEEK_END);
    // long lSize = ftell (pFile);

    rewind(pFile);

    lSize = num_bytes;

    if (lSize != num_bytes) {
      fprintf(stderr,
              "%s contains %lu bytes when %lu were expected. Exiting.\n",
              infile, lSize, num_bytes);
      return 1;
    }

    buffer = (double *)malloc(sizeof(char) * lSize);
    if (buffer == NULL) {
      fputs("Memory error", stderr);
      exit(2);
    }

    result = fread(buffer, 1, lSize, pFile);
    if (result != lSize) {
      fputs("Reading error", stderr);
      exit(3);
    }
    fclose(pFile);
  }

  double data_L_inf_norm = 0;
  for (int i = 0; i < num_doubles; ++i) {
    double temp = fabs(buffer[i]);
    if (temp > data_L_inf_norm)
      data_L_inf_norm = temp;
  }

  double *in_buff = (double *)malloc(sizeof(double) * num_doubles);
  memcpy(in_buff, buffer, sizeof(double) * num_doubles);

  int iflag = 1; // 0 -> float, 1 -> double
  int out_size;

  unsigned char *mgard_comp_buff;

  mgard_comp_buff =
      mgard_compress(iflag, in_buff, out_size, nrow, ncol, nfib, tol);
  free(in_buff);

  printf("In size:  %10ld  Out size: %10d  Compression ratio: %10ld \n", lSize,
         out_size, lSize / out_size);

  double *mgard_out_buff;
  double dummy = 0;
  mgard_out_buff = mgard_decompress(iflag, dummy, mgard_comp_buff, out_size,
                                    nrow, ncol, nfib);

  // FILE *qfile;
  // qfile = fopen ( outfile , "wb" );
  // result = fwrite (mgard_out_buff, 1, lSize, qfile);
  // fclose(qfile);
  // if (result != lSize) {fputs ("Writing error",stderr); exit (4);}

  double error_L_inf_norm = 0;
  for (int i = 0; i < num_doubles; ++i) {
    double temp = fabs(buffer[i] - mgard_out_buff[i]);
    if (temp > error_L_inf_norm)
      error_L_inf_norm = temp;
  }
  double relative_L_inf_error = error_L_inf_norm / data_L_inf_norm;

  // Maximum length (plus one for terminating byte) of norm name.
  size_t N = 10;
  char norm_name[N];
  int num_chars_written;
  if (isinf(s)) {
    num_chars_written = snprintf(norm_name, N, "L^infty");
  } else if (s == 0) {
    num_chars_written = snprintf(norm_name, N, "L^2");
  } else {
    num_chars_written = snprintf(norm_name, N, "H^%.1f", s);
  }
  if (num_chars_written <= 0 || num_chars_written >= N) {
    norm_name[0] = '?';
    norm_name[1] = 0;
  }
  printf("Rel. %s error tolerance: %10.5E \n", norm_name, tol);
  printf("Rel. L^infty error: %10.5E \n", relative_L_inf_error);

  if (relative_L_inf_error < tol) {
    printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    return 0;
  } else {
    printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
    return 1;
  }
}
