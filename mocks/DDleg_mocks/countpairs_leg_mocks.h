/* File: countpairs_mocks_leg.h */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#include "defs.h"//for struct config_options
#include <stdint.h> //for uint64_t

//define the results structure
  typedef struct{
    uint8_t *ells;
    double *supp;
    double *savg;
    uint64_t *npairs;
    double *poles;
    int nells;
    int nsbin;
  } results_countpairs_mocks_leg;

  extern int countpairs_mocks_leg(const int64_t ND1, void *X1, void *Y1, void *Z1,
                                  const int64_t ND2, void *X2, void *Y2, void *Z2,
                                  const int numthreads,
                                  const int autocorr,
                                  polearray *bins,
                                  double rmin,
                                  double rmax,
                                  double mumax,
                                  results_countpairs_mocks_leg *results,
                                  struct config_options *options,
                                  struct extra_options *extra) __attribute__((warn_unused_result));

  extern void free_results_mocks_leg(results_countpairs_mocks_leg *results);

#ifdef __cplusplus
}
#endif