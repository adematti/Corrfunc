/* File: countpairs_mocks_bessel.c */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "countpairs_bessel_mocks.h" //function proto-type for API
#include "countpairs_bessel_mocks_impl_double.h"//actual implementations for double
#include "countpairs_bessel_mocks_impl_float.h"//actual implementations for float


void free_results_mocks_bessel(results_countpairs_mocks_bessel *results)
{
    if(results == NULL)
        return;

    free(results->ells);results->ells = NULL;
    free(results->modes);results->modes = NULL;
    free(results->poles);results->poles = NULL;
    results->nells = 0;
    results->nmodes = 0;
}



int countpairs_mocks_bessel(const int64_t ND1, void * restrict X1, void * restrict Y1, void * restrict Z1,
                            const int64_t ND2, void * restrict X2, void * restrict Y2, void * restrict Z2,
                            const int numthreads,
                            const int autocorr,
                            polearray *bins,
                            double rmin,
                            double rmax,
                            double mumax,
                            results_countpairs_mocks_bessel *results,
                            struct config_options *options,
                            struct extra_options *extra)
{

    if( ! (options->float_type == sizeof(float) || options->float_type == sizeof(double))){
        fprintf(stderr,"ERROR: In %s> Can only handle doubles or floats. Got an array of size = %zu\n",
            __FUNCTION__, options->float_type);
        return EXIT_FAILURE;
    }

    if( strncmp(options->version, STR(VERSION), sizeof(options->version)/sizeof(char)-1) != 0) {
        fprintf(stderr,"Error: Do not know this API version = `%s'. Expected version = `%s'\n", options->version, STR(VERSION));
        return EXIT_FAILURE;
    }

    if(options->float_type == sizeof(float)) {
        return countpairs_mocks_bessel_float(ND1, (float *) X1, (float *) Y1, (float *) Z1,
                                             ND2, (float *) X2, (float *) Y2, (float *) Z2,
                                             numthreads,
                                             autocorr,
                                             bins,
                                             rmin,
                                             rmax,
                                             mumax,
                                             results,
                                             options,
                                             extra);
  } else {
        return countpairs_mocks_bessel_double(ND1, (double *) X1, (double *) Y1, (double *) Z1,
                                              ND2, (double *) X2, (double *) Y2, (double *) Z2,
                                              numthreads,
                                              autocorr,
                                              bins,
                                              rmin,
                                              rmax,
                                              mumax,
                                              results,
                                              options,
                                              extra);
    }

}
