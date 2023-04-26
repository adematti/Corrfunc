/* File: run_correlations_mocks.c */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

/*
  Example code to show how to use the correlation function libraries
  Author: Manodeep Sinha <manodeep@gmail.com>
  Date: At some point in early 2015.

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#include "function_precision.h"
#include "io.h"
#include "defs.h"
#include "utils.h"

/* Library proto-types + struct definitions in the ../..//include directory */
#include "countpairs_rp_pi_mocks.h"
#include "countpairs_s_mu_mocks.h"
#include "countspheres_mocks.h"

#ifndef MAXLEN
#define MAXLEN 500
#endif

void Printhelp(void);

void Printhelp(void)
{
    fprintf(stderr,ANSI_COLOR_RED "=========================================================================\n") ;
    fprintf(stderr,"   --- run_correlations_mocks file format binfile boxsize numthreads\n") ;
    fprintf(stderr,"   --- Measure the auto-correlation functions DD(r), DD(rp,pi) and vpf(r) for a single file\n");
    fprintf(stderr,"     * file         = name of data file\n") ;
    fprintf(stderr,"     * format       = format of data file  (a=ascii, f=fast-food)\n") ;
    fprintf(stderr,"     * binfile      = name of ascii file containing the r-bins (rmin rmax for each bin)\n") ;
    fprintf(stderr,"     * pimax        = pimax   (in same units as X/Y/Z of the data)\n");
    fprintf(stderr,"     * mu_max       = Max. value of the cosine of the angle to the LOS (must be within [0.0, 1.0])\n");
    fprintf(stderr,"     * nmu_bins     = Number of linear bins to create (the bins themselves range from [0.0, mu_max]\n");
#if defined(USE_OMP) && defined(_OPENMP)
    fprintf(stderr,"     * numthreads   = number of threads to use\n");
#endif
    fprintf(stderr,"=========================================================================" ANSI_COLOR_RESET "\n") ;
}

int main(int argc, char **argv)
{
    char file[MAXLEN];
    char fileformat[MAXLEN];
    char binfile[MAXLEN];
    DOUBLE *X1=NULL,*Y1=NULL,*Z1=NULL;
    struct timeval t0,t1;
    DOUBLE pimax;
    int nmu_bins;
    DOUBLE mu_max;
    int nthreads=4;

    struct config_options options = get_config_options();
    options.verbose=1;
    options.periodic=0;
    options.need_avg_sep=1;
    options.float_type = sizeof(*X1);

#if defined(_OPENMP)
    const char argnames[][30]={"file", "format", "binfile", "pimax", "mu_max", "nmu_bins", "numthreads"};
#else
    const char argnames[][30]={"file", "format", "binfile", "pimax", "mu_max", "nmu_bins"};
#endif
    int nargs=sizeof(argnames)/(sizeof(char)*30);

    if(argc > 1) {
        //Command-line options were supplied - check that they are correct
        if(argc < (nargs + 1) ) {
            //Not enough options were supplied
            Printhelp();
            return EXIT_FAILURE;
        } else {
            //Correct number of options - let's parse them.
            my_snprintf(file,MAXLEN, "%s",argv[1]);
            my_snprintf(fileformat,MAXLEN, "%s",argv[2]);
            my_snprintf(binfile,MAXLEN,"%s",argv[3]);
            pimax=atof(argv[4]);
            mu_max=atof(argv[5]);
            nmu_bins=atoi(argv[6]);
#if defined(_OPENMP)
            nthreads = atoi(argv[7]);
#endif
        }
    } else {
        my_snprintf(file, MAXLEN, "%s", "../tests/data/Mr19_mock_northonly.xyz.txt");
        my_snprintf(fileformat, MAXLEN, "%s","a");
        my_snprintf(binfile, MAXLEN,"%s","../tests/bins");
        pimax=40.0;
        mu_max=1.0;
        nmu_bins=10;
    }

    fprintf(stderr,ANSI_COLOR_BLUE  "Running `%s' with the parameters \n",argv[0]);
    fprintf(stderr,"\n\t\t -------------------------------------\n");
    fprintf(stderr,"\t\t %-10s = %s \n",argnames[0],file);
    fprintf(stderr,"\t\t %-10s = %s \n",argnames[1],fileformat);
    fprintf(stderr,"\t\t %-10s = %s \n",argnames[2],binfile);
    fprintf(stderr,"\t\t %-10s = %10.4lf\n",argnames[3],pimax);
    fprintf(stderr,"\t\t %-10s = %10.4lf\n",argnames[4],mu_max);
    fprintf(stderr,"\t\t %-10s = %dlf\n",argnames[5],nmu_bins);
#if defined(_OPENMP)
    fprintf(stderr,"\t\t %-10s = %d\n",argnames[6],nthreads);
#endif
    fprintf(stderr,"\t\t -------------------------------------" ANSI_COLOR_RESET "\n");

    binarray bins;
    read_binfile(binfile, &bins);

    //Read-in the data
    const int64_t ND1 = read_positions(file,fileformat,sizeof(*X1),3, &X1, &Y1, &Z1);

    int autocorr=1;
    DOUBLE *X2 = X1;
    DOUBLE *Y2 = Y1;
    DOUBLE *Z2 = Z1;
    int64_t ND2 = ND1;

    //Do the DD(rp, pi) counts
    {
        gettimeofday(&t0,NULL);
#if defined(_OPENMP)
        fprintf(stderr,ANSI_COLOR_MAGENTA "Command-line for running equivalent DD(rp,pi) calculation would be:\n `%s %s %s %s %s %s %lf %d'" ANSI_COLOR_RESET "\n",
                "../DDrppi_mocks/DDrppi_mocks",file,fileformat,file,fileformat,binfile,pimax,nthreads);
#else
        fprintf(stderr,ANSI_COLOR_MAGENTA "Command-line for running equivalent DD(rp,pi) calculation would be:\n `%s %s %s %s %s %s %lf'" ANSI_COLOR_RESET "\n",
                "../DDrppi_mocks/DDrppi_mocks",file,fileformat,file,fileformat,binfile,pimax);
#endif

        results_countpairs_mocks results;
        int status = countpairs_mocks(ND1,X1,Y1,Z1,
                                      ND2,X2,Y2,Z2,
                                      nthreads,
                                      autocorr,
                                      &bins,
                                      pimax,
                                      (int) pimax,
                                      &results,
                                      &options, NULL);
        if(status != EXIT_SUCCESS) {
            return status;
        }

        gettimeofday(&t1,NULL);
        double pair_time = ADD_DIFF_TIME(t0,t1);
#if 0
        const DOUBLE dpi = 2.*pimax/(DOUBLE)results.npibin ;
        const int npibin = results.npibin;
        for(int i=1;i<results.nbin;i++) {
            const double logrp = LOG10(results.rupp[i]);
            for(int j=0;j<npibin;j++) {
                int index = i*(npibin+1) + j;
                fprintf(stdout,"%10"PRIu64" %20.8lf %20.8lf  %20.8lf \n",results.npairs[index],results.rpavg[index],logrp,(j+1)*dpi-pimax);
            }
        }
#endif
        fprintf(stderr,ANSI_COLOR_GREEN "Done DD(rp,pi) auto-correlation. Ngalaxies = %12"PRId64" Time taken = %8.2lf seconds " ANSI_COLOR_RESET "\n", ND1, pair_time);


        //free the result structure
        free_results_mocks(&results);
    }



    //Do the DD(s, mu) counts
    {
        gettimeofday(&t0,NULL);
#if defined(_OPENMP)
        fprintf(stderr,ANSI_COLOR_MAGENTA "Command-line for running equivalent DD(s,mu) calculation would be:\n `%s %s %s %s %s %s %lf %d %d'"ANSI_COLOR_RESET"\n",
                "../DDsmu_mocks/DDsmu_mocks",file,fileformat,file,fileformat,binfile,mu_max,nmu_bins,nthreads);
#else
        fprintf(stderr,ANSI_COLOR_MAGENTA "Command-line for running equivalent DD(s,mu) calculation would be:\n `%s %s %s %s %s %s %lf %d'"ANSI_COLOR_RESET"\n",
                "../DDsmu_mocks/DDsmu_mocks",file,fileformat,file,fileformat,binfile,mu_max,nmu_bins);
#endif

        results_countpairs_mocks_s_mu results;
        int status = countpairs_mocks_s_mu(ND1,X1,Y1,Z1,
                                           ND2,X2,Y2,Z2,
                                           nthreads,
                                           autocorr,
                                           &bins,
                                           mu_max,
                                           nmu_bins,
                                           &results,
                                           &options, NULL);
        if(status != EXIT_SUCCESS) {
            return status;
        }

        gettimeofday(&t1,NULL);
        double pair_time = ADD_DIFF_TIME(t0,t1);
#if 0
        const DOUBLE dmu = 2.*mu_max/(DOUBLE)results.nmu_bins ;
        const int nmubin = results.nmu_bins;
        for(int i=1;i<results.nsbin;i++) {
            const double log_supp = LOG10(results.supp[i]);
            for(int j=0;j<nmubin;j++) {
                const int index = i*(nmubin+1) + j;
                fprintf(stdout,"%10"PRIu64" %20.8lf %20.8lf  %20.8lf %20.8lf \n",results.npairs[index],results.savg[index],log_supp,(j+1)*dmu-mu_max);
            }
        }

#endif
        fprintf(stderr,ANSI_COLOR_GREEN "Done DD(s,mu) auto-correlation. Ngalaxies = %12"PRId64" Time taken = %8.2lf seconds "ANSI_COLOR_RESET"\n", ND1, pair_time);

        //free the result structure
        free_results_mocks_s_mu(&results);
    }

    //Do the VPF
    {
        gettimeofday(&t0,NULL);
        const double rmax=10.0;
        const int nbin=10;
        const int nc=10000;
        const int num_pN=6;
        const int64_t Nran=nc;//Need to set it to nc so that the loop runs
        DOUBLE *xran=NULL,*yran=NULL,*zran=NULL;
        const int threshold_neighbors=1;
        const char centers_file[]="../tests/data/Mr19_centers_xyz_forVPF_rmax_10Mpc.txt";
        fprintf(stderr,ANSI_COLOR_MAGENTA "Command-line for running equivalent DD(theta) calculation would be:\n"
                "`%s %lf %d %d %d %lf %s %s %s %s %s'"ANSI_COLOR_RESET "\n",
                "../vpf_mocks/vpf_mocks",rmax,nbin,nc,num_pN,0.0,file,fileformat,"junk","junkformat",centers_file);

        results_countspheres_mocks results;
        int status = countspheres_mocks(ND1, X1, Y1, Z1,
                                        Nran, xran, yran, zran,
                                        threshold_neighbors,
                                        rmax, nbin, nc,
                                        num_pN,
                                        centers_file,
                                        &results,
                                        &options, NULL);
        if(status != EXIT_SUCCESS) {
            return status;
        }


        gettimeofday(&t1,NULL);
        double sphere_time = ADD_DIFF_TIME(t0,t1);

#if 0
        //Output the results
        const DOUBLE rstep = rmax/(DOUBLE)nbin ;
        for(int ibin=0;ibin<results.nbin;ibin++) {
            const double r=(ibin+1)*rstep;
            fprintf(stdout,"%10.2"REAL_FORMAT" ", r);
            for(int i=0;i<num_pN;i++) {
                fprintf(stdout," %10.4e", (results.pN)[ibin][i]);
            }
            fprintf(stdout,"\n");
        }
#endif
        fprintf(stderr,ANSI_COLOR_GREEN "Done VPF. Ngalaxies = %12"PRId64" Time taken = %8.2lf seconds" ANSI_COLOR_RESET "\n", ND1, sphere_time);
        free_results_countspheres_mocks(&results);
    }



    free(X1);free(Y1);free(Z1);
    free_binarray(&bins);
    return EXIT_SUCCESS;
}
