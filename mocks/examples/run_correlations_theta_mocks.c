/* File: run_correlations_theta_mocks.c */
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
#include "countpairs_theta_mocks.h"

#ifndef MAXLEN
#define MAXLEN 500
#endif

void Printhelp(void);

void Printhelp(void)
{
    fprintf(stderr,ANSI_COLOR_RED "=========================================================================\n") ;
    fprintf(stderr,"   --- run_correlations_mocks file format binfile boxsize numthreads\n") ;
    fprintf(stderr,"   --- Measure the auto-correlation functions DD(theta) for a single file\n");
    fprintf(stderr,"     * file         = name of data file\n") ;
    fprintf(stderr,"     * format       = format of data file  (a=ascii, f=fast-food)\n") ;
    fprintf(stderr,"     * binfile      = name of ascii file containing the theta-bins\n") ;
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
    DOUBLE *ra1=NULL,*dec1=NULL;
    struct timeval t0,t1;
    int nthreads=4;

    struct config_options options = get_config_options();
    options.verbose=1;
    options.periodic=0;
    options.need_avg_sep=1;
    options.float_type = sizeof(*ra1);

#if defined(_OPENMP)
    const char argnames[][30]={"file", "format", "binfile", "numthreads"};
#else
    const char argnames[][30]={"file", "format", "binfile"};
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
#if defined(_OPENMP)
            nthreads = atoi(argv[4]);
#endif
        }
    } else {
        my_snprintf(file, MAXLEN, "%s", "../tests/data/Mr19_mock_northonly.rdcz.txt");
        my_snprintf(fileformat, MAXLEN, "%s","a");
        my_snprintf(binfile, MAXLEN,"%s","../tests/angular_bins");
    }

    fprintf(stderr,ANSI_COLOR_BLUE  "Running `%s' with the parameters \n",argv[0]);
    fprintf(stderr,"\n\t\t -------------------------------------\n");
    fprintf(stderr,"\t\t %-10s = %s \n",argnames[0],file);
    fprintf(stderr,"\t\t %-10s = %s \n",argnames[1],fileformat);
    fprintf(stderr,"\t\t %-10s = %s \n",argnames[2],binfile);
#if defined(_OPENMP)
    fprintf(stderr,"\t\t %-10s = %d\n",argnames[3],nthreads);
#endif
    fprintf(stderr,"\t\t -------------------------------------" ANSI_COLOR_RESET "\n");

    binarray bins;
    read_binfile(binfile, &bins);

    //Read-in the data
    const int64_t ND1 = read_positions(file,fileformat,sizeof(*ra1), 2, &ra1, &dec1);

    int autocorr=1;
    DOUBLE *ra2 = ra1;
    DOUBLE *dec2 = dec1;
    int64_t ND2 = ND1;

    //Do the DD(theta) counts
    {
        gettimeofday(&t0,NULL);
#if defined(_OPENMP)
        fprintf(stderr,ANSI_COLOR_MAGENTA "Command-line for running equivalent w(theta) calculation would be:\n `%s %s %s %s %s %s %d'" ANSI_COLOR_RESET "\n",
                "../DDtheta_mocks/DDtheta_mocks",file,fileformat,file,fileformat,binfile,nthreads);
#else
        fprintf(stderr,ANSI_COLOR_MAGENTA "Command-line for running equivalent w(theta) calculation would be:\n `%s %s %s %s %s %s '" ANSI_COLOR_RESET "\n",
                "../DDtheta_mocks/DDtheta_mocks",file,fileformat,file,fileformat,binfile);
#endif

        results_countpairs_theta results;
        options.fast_acos=1;//over-ride Makefile option
        int status = countpairs_theta_mocks(ND1,ra1,dec1,
                                            ND2,ra2,dec2,
                                            nthreads,
                                            autocorr,
                                            &bins,
                                            &results,
                                            &options, NULL);
        if(status != EXIT_SUCCESS) {
            return status;
        }
        gettimeofday(&t1,NULL);
        DOUBLE pair_time = ADD_DIFF_TIME(t0,t1);

#if 0
        /*---Output-Pairs-------------------------------------*/
        DOUBLE theta_low = results.theta_upp[0];
        for(int i=1;i<results.nbin;i++) {
            fprintf(stdout,"%10"PRIu64" %20.8lf %20.8lf %20.8lf \n",results.npairs[i],results.theta_avg[i],theta_low,results.theta_upp[i]);
            theta_low=results.theta_upp[i];
        }
#endif
        fprintf(stderr,ANSI_COLOR_GREEN "Done wtheta. Ngalaxies = %12"PRId64" Time taken = %8.2lf seconds" ANSI_COLOR_RESET "\n", ND1, pair_time);

        //free the result structure
        free_results_countpairs_theta(&results);
    }


    free(ra1);free(dec1);
    free_binarray(&bins);
    return EXIT_SUCCESS;
}
