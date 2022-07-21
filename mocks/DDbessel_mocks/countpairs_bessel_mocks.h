/* File: countpairs_mocks_bessel.h */
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
    double *modes;
    double *poles;
    int nells;
    int nmodes;
  } results_countpairs_mocks_bessel;

  extern int countpairs_mocks_bessel(const int64_t ND1, void *X1, void *Y1, void *Z1,
                                     const int64_t ND2, void *X2, void *Y2, void *Z2,
                                     const int numthreads,
                                     const int autocorr,
                                     polearray *bins,
                                     double rmin,
                                     double rmax,
                                     double mumax,
                                     results_countpairs_mocks_bessel *results,
                                     struct config_options *options,
                                     struct extra_options *extra) __attribute__((warn_unused_result));

  extern void free_results_mocks_bessel(results_countpairs_mocks_bessel *results);

#ifdef __cplusplus
}
#endif
