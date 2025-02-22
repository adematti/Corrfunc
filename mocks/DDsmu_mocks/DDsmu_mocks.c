/* File: DDsmu_mocks.c */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

/* PROGRAM DDsmu

--- DDsmu file1 format1 file2 format2 sbinfile Nmu numthreads [weight_method weights_file1 weights_format1 [weights_file2 weights_format2]] > DDfile
--- Measure the cross-correlation function xi(rp,pi) for two different
   data files (or autocorrelation if data1=data2).
 * file1         = name of first data file
 * format1       = format of first data file  (a=ascii, c=csv, f=fast-food)
 * file2         = name of second data file
 * format2       = format of second data file (a=ascii, c=csv, f=fast-food)
 * sbinfile       = name of ascii file containing the r-bins (rmin rmax for each bin)
 * mu_max        = maximum mu value (>0 and <= 1.0)
 * nmu_bins      = number of mu bins
 * numthreads    = number of threads to use
--- OPTIONAL ARGS:
 * weight_method = the type of pair weighting to apply. Options are: 'pair_product', 'inverse_bitwise', 'none'. Default: 'none'.
 * pair_weights = name of file containing the angular upweights (costheta, weight)
 * weights_file1 = name of file containing the weights corresponding to the first data file
 * weights_format1 = format of file containing the weights corresponding to the first data file
 * weights_file2 = name of file containing the weights corresponding to the second data file
 * weights_format2 = format of file containing the weights corresponding to the second data file
---OUTPUT:
 > DDfile        = name of output file. Contains <npairs savg logrp pi weightavg>

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>

#include "defs.h" //for basic API + all macros
#include "function_precision.h" //definition of DOUBLE
#include "countpairs_s_mu_mocks.h" //function proto-type for countpairs
#include "io.h" //function proto-type for file input
#include "utils.h" //general utilities


void Printhelp(void);

static int get_number_of_columns(const char *filename) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int ncols = 0;

    fp = fopen(filename, "r");
    if (fp == NULL) return ncols;

    while ((read = getline(&line, &len, fp)) != -1) {
        bool word = false;
        for (size_t ic=0; ic<len; ic++) {
            if ((line[ic] == ' ') || (line[ic] == ',')) word = false;
            else if (!word) {
                ncols++;
                word = true;
            }
        }
        break;
    }
    free(line);
    fclose(fp);
    return ncols;
}

int main(int argc, char *argv[])
{
    /*---Arguments-------------------------*/
    char *file1=NULL,*file2=NULL, *weights_file1=NULL,*weights_file2=NULL;
    char *fileformat1=NULL,*fileformat2=NULL, *weights_fileformat1=NULL,*weights_fileformat2=NULL;
    char *sbinfile=NULL;
    char *weight_method_str=NULL;
    int nmu_bins;
    DOUBLE mu_max;

    weight_method_t weight_method = NONE;
    weight_type_t weight_types[MAX_NUM_WEIGHTS];
    int num_weights = 0;
    bool use_gpu = false;

    /*---Data-variables--------------------*/
    int64_t ND1,ND2 ;

    DOUBLE *X1,*Y1,*Z1, *weights1[MAX_NUM_WEIGHTS] = {NULL};
    DOUBLE *X2,*Y2,*Z2, *weights2[MAX_NUM_WEIGHTS] = {NULL};
    DOUBLE *pair_weights[2] = {NULL};

    struct timeval t_end,t_start,t0,t1;
    double read_time=0.0;
    gettimeofday(&t_start,NULL);
    int nthreads=1;

    /*---Corrfunc-variables----------------*/
#if defined(_OPENMP)
    char argnames[][30]={"file1","format1","file2","format2","sbinfile","mu_max","nmu_bins","numthreads"};
#else
    char argnames[][30]={"file1","format1","file2","format2","sbinfile","mu_max","nmu_bins"};
#endif
    char optargnames[][30]={"weight_method","pair_weights","weights_file1","weights_format1","weights_file2","weights_format2"};

    int nargs=sizeof(argnames)/(sizeof(char)*30);
    int noptargs=sizeof(optargnames)/(sizeof(char)*30);

    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i],"-gpu")) {
            use_gpu = true;
            for (int j = i+1; j < argc; j++) {
              argv[j-1] = argv[j];
            }
            argv[argc-1] = "-gpu";
        }
    }
    if (use_gpu) fprintf(stderr, "USE GPU\n");

    /* Validate optional arguments */
    int noptargs_given = argc - (nargs + 1);
    if (use_gpu) noptargs_given--;
    int with_pair_weights = (int) ((noptargs_given == 4) || (noptargs_given == 6));
    if (with_pair_weights) noptargs_given--;
    else {
        for (int i=1; i<noptargs-1; i++) strcpy(optargnames[i], optargnames[i+1]);  // shift all arguments
        strcpy(optargnames[noptargs-1], "");
    }

    /*---Read-arguments-----------------------------------*/
    if(argc< (nargs+1)) {
        Printhelp() ;
        fprintf(stderr,"\nFound: %d parameters\n ",argc-1);
        int i;
        for(i=1;i<argc;i++) {
            if(i <= nargs)
                fprintf(stderr,"\t\t %s = `%s' \n",argnames[i-1],argv[i]);
            else if(i <= nargs + noptargs) {
                fprintf(stderr,"\t\t %s = `%s' \n",optargnames[i-1-nargs],argv[i]);
            }
            else
                fprintf(stderr,"\t\t <> = `%s' \n",argv[i]);
        }
        fprintf(stderr,"\nMissing required parameters \n");
        for(i=argc;i<=nargs;i++)
            fprintf(stderr,"\t\t %s = `?'\n",argnames[i-1]);
        return EXIT_FAILURE;
    }

    if(noptargs_given != 0 && noptargs_given != 3 && noptargs_given != 5){
        Printhelp();
        fprintf(stderr,"\nFound: %d optional arguments; must be 0 (no weights), 3 or 4 (for one set of weights) or 5 or 6 (for two sets)\n ", noptargs_given);
        int i;
        for(i=nargs+1;i<argc;i++) {
            if(i <= nargs + noptargs)
                fprintf(stderr,"\t\t %s = `%s' \n",optargnames[i-nargs-1],argv[i]);
            else
                fprintf(stderr,"\t\t <> = `%s' \n",argv[i]);
        }
        return EXIT_FAILURE;
    }

    file1=argv[1];
    fileformat1=argv[2];
    file2=argv[3];
    fileformat2=argv[4];
    sbinfile=argv[5];

    mu_max=1.0;
    sscanf(argv[6],"%"REAL_FORMAT,&mu_max);
    nmu_bins=-10;
    sscanf(argv[7],"%d",&nmu_bins);

#if defined(USE_OMP) && defined(_OPENMP)
    nthreads=atoi(argv[8]);
    assert(nthreads >= 1 && "Number of threads must be at least 1");
#endif

    int64_t Npw = 0;
    int iarg = 0;
    if(noptargs_given >= 3){
       weight_method_str = argv[nargs + 1 + iarg];
       int wstatus = get_weight_method_by_name(weight_method_str, &weight_method);
       if(wstatus != EXIT_SUCCESS){
         fprintf(stderr, "Error: Unknown weight method \"%s\"\n", weight_method_str);
         return EXIT_FAILURE;
       }
       num_weights = get_num_weights_by_method(weight_method);
       if (with_pair_weights) {
        if (weight_method == INVERSE_BITWISE) {
            Npw = read_columns_into_array(argv[nargs + 2 + iarg], "a", sizeof(DOUBLE), 2, (void **) pair_weights);
            iarg += 1;
        }
        else {
            fprintf(stderr, "Error: pair weights not accepted with weight method \"%s\"\n", weight_method_str);
            return EXIT_FAILURE;
        }
       }
       weights_file1 = argv[nargs + 2 + iarg];
       weights_fileformat1 = argv[nargs + 3 + iarg];

       if (weight_method == INVERSE_BITWISE) {
         num_weights = get_number_of_columns(weights_file1);
         //num_weights = 5;  // hard-coded, 4 bitwise weights and 1 float weights
         for (int ii=0; ii<num_weights - 1; ii++) weight_types[ii] = INT_TYPE;
         weight_types[num_weights - 1] = FLOAT_TYPE;
       }
    }
    if(noptargs_given >= 5){
       weights_file2 = argv[nargs + 4 + iarg];
       weights_fileformat2 = argv[nargs + 5 + iarg];
    }

    int autocorr=0;
    if(strcmp(file1,file2)==0) {
        autocorr=1;
    }

    fprintf(stderr,"Running `%s' with the parameters \n",argv[0]);
    fprintf(stderr,"\n\t\t -------------------------------------\n");
    for(int i=1;i<argc;i++) {
        if(i <= nargs) {
            fprintf(stderr,"\t\t %-10s = %s \n",argnames[i-1],argv[i]);
        } else if (i > nargs) {
            if (i <= nargs + noptargs_given + with_pair_weights){
                fprintf(stderr,"\t\t %-10s = %s \n",optargnames[i - nargs - 1],argv[i]);
            } else {
                fprintf(stderr,"\t\t <> = `%s' \n",argv[i]);
            }
        }
    }
    fprintf(stderr,"\t\t -------------------------------------\n");

    binarray bins;
    read_binfile(sbinfile, &bins);
    /*---Read-data1-file----------------------------------*/
    gettimeofday(&t0,NULL);
    ND1=read_positions(file1,fileformat1,sizeof(DOUBLE), 3, &X1, &Y1, &Z1);
    gettimeofday(&t1,NULL);
    read_time += ADD_DIFF_TIME(t0,t1);
    gettimeofday(&t0,NULL);

    /* Read weights file 1 */
    if(weights_file1 != NULL){
        gettimeofday(&t0,NULL);
        int64_t wND1 = read_columns_into_array(weights_file1,weights_fileformat1, sizeof(DOUBLE), num_weights, (void **) weights1);
        gettimeofday(&t1,NULL);
        read_time += ADD_DIFF_TIME(t0,t1);

        if(wND1 != ND1){
          fprintf(stderr, "Error: read %"PRId64" lines from %s, but read %"PRId64" from %s\n", wND1, weights_file1, ND1, file1);
          return EXIT_FAILURE;
        }
    }

    if (autocorr==0) {
        /*---Read-data2-file----------------------------------*/
        ND2=read_positions(file2,fileformat2,sizeof(DOUBLE), 3, &X2, &Y2, &Z2);
        gettimeofday(&t1,NULL);
        read_time += ADD_DIFF_TIME(t0,t1);

        if(weights_file2 != NULL){
            gettimeofday(&t0,NULL);
            int64_t wND2 = read_columns_into_array(weights_file2,weights_fileformat2, sizeof(DOUBLE), num_weights, (void **) weights2);
            gettimeofday(&t1,NULL);
            read_time += ADD_DIFF_TIME(t0,t1);

            if(wND2 != ND2){
              fprintf(stderr, "Error: read %"PRId64" lines from %s, but read %"PRId64" from %s\n", wND2, weights_file2, ND2, file2);
              return EXIT_FAILURE;
            }
        }
    } else {
        //None of these are required. But I prefer to preserve the possibility
        ND2 = ND1;
        X2 = X1;
        Y2 = Y1;
        Z2 = Z1;
        for(int w = 0; w < MAX_NUM_WEIGHTS; w++){
          weights2[w] = weights1[w];
        }
    }

    /*---Count-pairs--------------------------------------*/
    results_countpairs_mocks_s_mu results;
    struct config_options options = get_config_options();
    //set_selection_struct(&(options.selection), RP_SELECTION, 0, 25); //uncomment this line to test RP_SELECTION
    //options.instruction_set = FALLBACK;

    set_gpu_mode(&options, (uint8_t)use_gpu);

    /* Pack weights into extra options */
    struct extra_options extra = get_extra_options(weight_method);
    for(int w = 0; w < num_weights; w++){
        extra.weights0.weights[w] = (void *) weights1[w];
        extra.weights1.weights[w] = (void *) weights2[w];
    }
    if (weight_method == INVERSE_BITWISE) {
        set_weight_struct(&(extra.weights0), weight_method, weight_types, num_weights);
        set_weight_struct(&(extra.weights1), weight_method, weight_types, num_weights);
        if (Npw) set_pair_weight_struct(&(extra.pair_weight), pair_weights[0], pair_weights[1], Npw, 1, 0.);
    }

    int status = countpairs_mocks_s_mu(ND1,X1,Y1,Z1,
                                      ND2,X2,Y2,Z2,
                                      nthreads,
                                      autocorr,
                                      &bins,
                                      mu_max,
                                      nmu_bins,
                                      &results,
                                      &options,
                                      &extra);

    free(X1);free(Y1);free(Z1);
    for(int w = 0; w < num_weights; w++){
        free(weights1[w]);
    }
    if(autocorr == 0) {
        free(X2);free(Y2);free(Z2);
        for(int w = 0; w < num_weights; w++){
          free(weights2[w]);
        }
    }
    free_binarray(&bins);

    if(status != EXIT_SUCCESS) {
        return status;
    }

    const DOUBLE dmu = 2.*mu_max/(DOUBLE) results.nmu_bins;
    const int nmubin = results.nmu_bins;
    for(int i=1;i<results.nsbin;i++) {
        const double log_supp = LOG10(results.supp[i]);
        for(int j=0;j<nmubin;j++) {
            const int index = i*(nmubin+1) + j;
            fprintf(stdout,"%10"PRIu64" %20.8lf %20.8lf %20.8lf %20.12lf \n",results.npairs[index],results.savg[index],log_supp,(j+1)*dmu-mu_max, results.weightavg[index]);
        }
    }

    free_results_mocks_s_mu(&results);
    gettimeofday(&t_end,NULL);
    fprintf(stderr,"DDsmu> Done -  ND1=%"PRId64" ND2=%"PRId64". Time taken = %6.2lf seconds, read-in time = %6.2lf seconds \n",ND1,ND2,ADD_DIFF_TIME(t_start,t_end),read_time);
    return EXIT_SUCCESS;
}

/*---Print-help-information---------------------------*/
void Printhelp(void)
{
    fprintf(stderr,"=========================================================================\n") ;
#if defined(USE_OMP) && defined(_OPENMP)
    fprintf(stderr,"   --- DDsmu file1 format1 file2 format2 sbinfile mu_max nmu_bins numthreads [-gpu] [weight_method [pair_weights] weights_file1 weights_format1 [weights_file2 weights_format2]] > DDfile\n") ;
#else
    fprintf(stderr,"   --- DDsmu file1 format1 file2 format2 sbinfile mu_max nmu_bins [-gpu] [weight_method [pair_weights] weights_file1 weights_format1 [weights_file2 weights_format2]] > DDfile\n") ;
#endif
    fprintf(stderr,"   --- Measure the cross-correlation function xi(rp,pi) for two different\n") ;
    fprintf(stderr,"       data files (or autocorrelation if data1=data2).\n") ;
    fprintf(stderr,"     * data1         = name of first data file\n") ;
    fprintf(stderr,"     * format1       = format of first data file  (a=ascii, c=csv, f=fast-food)\n") ;
    fprintf(stderr,"     * data2         = name of second data file\n") ;
    fprintf(stderr,"     * format2       = format of second data file (a=ascii, c=csv, f=fast-food)\n") ;
    fprintf(stderr,"     * sbinfile      = name of ascii file containing the r-bins (rmin rmax for each bin)\n") ;
    fprintf(stderr,"     * mu_max        = maximum mu value (>0 and <= 1.0)\n") ;
    fprintf(stderr,"     * nmu_bins      = number of mu bins\n") ;
#if defined(USE_OMP) && defined(_OPENMP)
    fprintf(stderr,"     * numthreads    = number of threads to use\n");
#endif
    fprintf(stderr,"   --- OPTIONAL ARGS:\n");
    fprintf(stderr,"     * weight_method = the type of pair weighting to apply. Options are: 'pair_product', 'inverse_bitwise', 'none'. Default: 'none'.\n");
    fprintf(stderr,"     * pair_weights = name of file containing the angular upweights (costheta, weight)\n");
    fprintf(stderr,"     * weights_file1 = name of file containing the weights corresponding to the first data file\n");
    fprintf(stderr,"     * weights_format1 = format of file containing the weights corresponding to the first data file\n");
    fprintf(stderr,"     * weights_file2 = name of file containing the weights corresponding to the second data file\n");
    fprintf(stderr,"     * weights_format2 = format of file containing the weights corresponding to the second data file\n");
    fprintf(stderr,"   ---OUTPUT:\n") ;
    fprintf(stderr,"     > DDfile        = name of output file. Contains <npairs savg logrp mu weightavg>\n") ;

    fprintf(stderr,"\n\tCompile options: \n");

#ifdef OUTPUT_SAVG
    fprintf(stderr,"Output SAVG = True\n");
#else
    fprintf(stderr,"Output SAVG = False\n");
#endif

#ifdef DOUBLE_PREC
    fprintf(stderr,"Precision = double\n");
#else
    fprintf(stderr,"Precision = float\n");
#endif

#if defined(USE_AVX) && defined(__AVX__)
    fprintf(stderr,"Use AVX = True\n");
#else
    fprintf(stderr,"Use AVX = False\n");
#endif

#if defined(USE_OMP) && defined(_OPENMP)
    fprintf(stderr,"Use OMP = True\n");
#else
    fprintf(stderr,"Use OMP = False\n");
#endif

#ifdef GPU
    fprintf(stderr,"Use GPU = True\n");
#else
    fprintf(stderr,"Use GPU = False\n");
#endif

    fprintf(stderr,"=========================================================================\n") ;
}
