/* File: _countpairs.c */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Now, include the numpy header*/
#include <numpy/arrayobject.h>

//for correlation functions
#include "countpairs.h"
#include "countpairs_rp_pi.h"
#include "countpairs_wp.h"
#include "countpairs_xi.h"
#include "countpairs_s_mu.h"

//for the vpf
#include "countspheres.h"

//for the instruction set detection
#include "cpu_features.h"

//for unicode characters
#include "macros.h"
#include "defs.h"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
//python3 follows
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#define INITERROR return NULL
PyObject *PyInit__countpairs(void);

#else
//python2 follows
#define GETSTATE(m) (&_state)
static struct module_state _state;
#define INITERROR return
PyMODINIT_FUNC init_countpairs(void);

#endif

#define NOTYPE_DESCR     (PyArray_DescrFromType(NPY_NOTYPE))

//File-scope variable
static int highest_isa;

//Docstrings for the methods
static char module_docstring[]             =    "Python extensions for calculating clustering statistics on simulations.\n"
    "\n"
    "countpairs       : Calculate the 3-D xi auto/cross-correlation function given two sets of arrays with Cartesian XYZ positions.\n"
    "countpairs_rp_pi : Calculate the 2-D DD("RP_CHAR","PI_CHAR") auto/cross-correlation function given two sets of arrays with Cartesian XYZ positions.\n"
    "countpairs_wp    : Calculate the projected auto-correlation function wp (assumes PERIODIC) given one set of arrays with Cartesian XYZ positions\n"
    "countpairs_xi    : Calculate the 3-d auto-correlation function xi (assumes PERIODIC) given one set of arrays with Cartesian XYZ positions\n"
    "countpairs_s_mu  : Calculate the 2-D DD(s,"MU_CHAR") auto/cross-correlation function given two sets of arrays with Cartesian XYZ positions.\n"
    "countpairs_vpf   : Calculate the counts-in-spheres given one set of arrays with Cartesian XYZ positions\n"
    "\n"
    "See `Corrfunc/call_correlation_functions.py` for example calls to each function in the extension.\n";

/* static char error_out_docstring[]          =  "Error-handler for the module."; */

/* function proto-type*/
static PyObject *countpairs_countpairs(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countpairs_rp_pi(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countpairs_wp(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countpairs_xi(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countpairs_s_mu(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countspheres_vpf(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_error_out(PyObject *module, const char *msg);

/* Inline documentation for the methods so that help(function) has something reasonably useful*/
static PyMethodDef module_methods[] = {
    /* {"countpairs_error_out"  ,(PyCFunction) countpairs_error_out        ,METH_VARARGS, error_out_docstring}, */
    {"countpairs"            ,(PyCFunction)(void(*)(void)) countpairs_countpairs       ,METH_VARARGS | METH_KEYWORDS,
     "countpairs(autocorr, nthreads, binfile, X1, Y1, Z1, weights1=None, weight_type=None, periodic=True,\n"
     "           X2=None, Y2=None, Z2=None, weights2=None, verbose=False, boxsize=0.0,\n"
     "           output_ravg=False, xbin_refine_factor=2, ybin_refine_factor=2,\n"
     "           zbin_refine_factor=1, max_cells_per_dim=100, copy_particles=True,\n"
     "           enable_min_sep_opt=True, c_api_timer=False, isa=-1)\n"
     "\n"
     "Calculate the 3-D pair-counts, "XI_CHAR"(r), auto/cross-correlation \n"
     "function given two sets of points represented by X1/Y1/Z1 and X2/Y2/Z2 \n"
     "arrays.\n\n"

     "Note, that this module only returns pair counts and not the actual \n"
     "correlation function "XI_CHAR"(r). See the mocks/wtheta/wtheta.c for \n"
     "computing "XI_CHAR"(r) from the output of DD(r). Also note that the \n"
     "python wrapper for this extension: `Corrfunc.theory.DD` is more user-friendly.\n"
     UNICODE_WARNING
     "\n"
     "Parameters \n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n\n"

     "autocorr : boolean\n"
     "    Boolean flag for auto/cross-correlation. If autocorr is set to 1,\n"
     "    are not used (but must still be passed, perhaps again as X1/Y1/Z1).\n\n"

     "nthreads : integer\n"
     "    The number of OpenMP threads to use. Has no effect if OpenMP was not\n"
     "    enabled during library compilation.\n\n"

     "binfile : string\n"
     "    Filename specifying the ``r`` bins for ``DD``. The file should\n"
     "    contain white-space separated values  of (rmin, rmax)  for each\n"
     "    ``r`` wanted. The bins do not need to be contiguous but must be in\n"
     "    increasing order (smallest bins come first). \n\n"

     "X1/Y1/Z1 : array-like, real (float/double)\n"
     "    The array of X/Y/Z positions for the first set of points.\n"
     "    Calculations are done in the precision of the supplied arrays.\n\n"

     "weights1 : array-like, real (float/double), shape (n_weights_per_particle,n_particles), optional\n"
     "    Weights for computing a weighted pair count.\n\n"

     "weight_type : str, optional\n"
     "    The type of pair weighting to apply.\n"
     "    Options: \"pair_product\", None\n"
     "    Default: None.\n\n"

     "periodic : boolean\n"
     "    Boolean flag to indicate periodic boundary conditions.\n\n"

     "X2/Y2/Z2 : array-like, real (float/double)\n"
     "    Array of XYZ positions for the second set of points. *Must* be the same\n"
     "    precision as the X1/Y1/Z1 arrays. Only required when ``autocorr==0``.\n\n"

     "weights2\n : array-like, real (float/double), shape (n_weights_per_particle,n_particles), optional\n"
     "    Weights for computing a weighted pair count.\n\n"

     "verbose : boolean (default false)\n"
     "    Boolean flag to control output of informational messages\n\n"

     "boxsize : 3-tuple of double\n"
     "    The (X,Y,Z) side lengths of the spatial domain.\n"
     "    Present to facilitate exact calculations for periodic wrapping.\n"
     "    If the boxsize in a dimension is 0., then\n"
     "    then that dimension's wrap is done based on the extent of the particle\n"
     "    distribution. If the boxsize in a dimension is -1., then periodicity\n"
     "    is disabled for that dimension.\n\n"

     "output_ravg : boolean (default false)\n"
     "    Boolean flag to output the average ``r`` for each bin. Code will\n"
     "    run slower if you set this flag. Also, note, if you are calculating\n"
     "    in single-precision, ``ravg`` will suffer from numerical loss of\n"
     "    precision and can not be trusted. If you need accurate ``ravg``\n"
     "    values, then pass in double precision arrays for the particle positions.\n\n"

     "(xyz)bin_refine_factor: integer (default (2,2,1) typical values in [1-3]) \n"
     "    Controls the refinement on the cell sizes. Can have up to a 20% impact \n"
     "    on runtime. \n\n"

     "max_cells_per_dim: integer (default 100, typical values in [50-300]) \n"
     "    Controls the maximum number of cells per dimension. Total number of cells \n"
     "    can be up to (max_cells_per_dim)^3. Only increase if ``rmax`` is too small \n"
     "    relative to the boxsize (and increasing helps the runtime). \n\n"

     "copy_particles: boolean (default True)\n"
     "    Boolean flag to make a copy of the particle positions\n"
     "    If set to False, the particles will be re-ordered in-place\n\n"

     "enable_min_sep_opt: boolean (default true)\n"
     "    Boolean flag to allow optimizations based on min. separation between pairs \n"
     "    of cells. Here to allow for comparison studies.\n\n"

     "c_api_timer : boolean (default false)\n"
     "    Boolean flag to measure actual time spent in the C libraries. Here\n"
     "    to allow for benchmarking and scaling studies.\n\n"

     "isa : integer (default -1)\n"
     "    Controls the runtime dispatch for the instruction set to use. Possible\n"
     "    options are: [-1, AVX512F, AVX, SSE42, FALLBACK]\n\n"
     "    Setting isa to -1 will pick the fastest available instruction\n"
     "    set on the current computer. However, if you set ``isa`` to, say,\n"
     "    ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "    revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "    Unless you are benchmarking the different instruction sets, you should\n"
     "    always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "    then the integer values correspond to the ``enum`` for the instruction set\n"
     "    defined in ``utils/defs.h``.\n\n"

     "Returns\n"
     "--------\n\n"
     "A tuple (results, time) \n\n"

     "results : A python list\n"
     "    A python list containing [rmin, rmax, ravg, npairs, weight_avg] for each radial bin\n"
     "    specified in the ``binfile``. If ``output_ravg`` is not set, then ``ravg``\n"
     "    will be set to 0.0 for all bins; similarly for ``weight_avg``. ``npairs`` contains the number of pairs\n"
     "    in that bin and can be used to compute the actual "XI_CHAR"(r) by\n"
     "    combining with (DR, RR) counts.\n\n"

     "time : double\n"
     "    if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "    in the API; otherwise time is set to 0.0\n\n"

     "Example\n"
     "-------\n\n"

     ">>> from Corrfunc._countpairs import countpairs\n"
     ">>> from Corrfunc.io import read_catalog\n"
     ">>> x,y,z = read_catalog()\n"
     ">>> autocorr=1\n"
     ">>> nthreads=2\n"
     ">>> (DD, time) = countpairs(autocorr, nthreads, '../tests/bins',x, y, z, \n"
     "                            X2=x, Y2=y, Z2=z,verbose=True)\n"
     "\n"
    },
    {"countpairs_rp_pi"      ,(PyCFunction)(void(*)(void)) countpairs_countpairs_rp_pi ,METH_VARARGS | METH_KEYWORDS,
     "countpairs_rp_pi(autocorr, nthreads, pimax, binfile, X1, Y1, Z1, weights1=None, weight_type=None,\n"
     "                 periodic=True, X2=None, Y2=None, Z2=None, weights2=None, verbose=False,\n"
     "                 boxsize=0.0, output_rpavg=False, xbin_refine_factor=2, ybin_refine_factor=2,\n"
     "                 zbin_refine_factor=1, max_cells_per_dim=100, copy_particles=True,\n"
     "                 enable_min_sep_opt=True, c_api_timer=False, isa=-1)\n"
     "\n"
     "Calculate the 3-D pair-counts corresponding to the real-space correlation\n"
     "function, "XI_CHAR"("RP_CHAR", "PI_CHAR") or wp("RP_CHAR"). Pairs which are separated\n"
     "by less than the ``rp`` bins (specified in ``binfile``) in the X-Y plane, and\n"
     "less than ``pimax`` in the Z-dimension are counted.\n\n"

     "Note, that this module only returns pair counts and not the actual\n"
     "correlation function "XI_CHAR"("RP_CHAR", "PI_CHAR"). See ``theory/DDrppi/wprp.c``\n"
     "for computing wp("RP_CHAR") from the pair counts returned by this module.\n"
     "Also note that the python wrapper for this extension: `Corrfunc.theory.DDrppi`\n"
     "is more user-friendly.\n"
     UNICODE_WARNING
     "\n"
     "Parameters\n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n\n"

     "autocorr: boolean, required\n"
     "    Boolean flag for auto/cross-correlation. If autocorr is set to 1,\n"
     "    are not used (but must still be passed, perhaps again as X1/Y1/Z1).\n\n"

     "nthreads: integer\n"
     "    The number of OpenMP threads to use. Has no effect if OpenMP was not\n"
     "    enabled during library compilation.\n\n"

     "pimax: double\n"
     "    A double-precision value for the maximum separation along\n"
     "    the Z-dimension. Note that only pairs with ``0 <= dz < pimax``\n"
     "    are counted (no equality).\n"
     "    Distances along the Z direction ("PI_CHAR") are binned with unit\n"
     "    depth. For instance, if ``pimax=40``, then 40 bins will be created\n"
     "    along the ``"PI_CHAR"`` direction.\n\n"

     "binfile : string\n"
     "    Filename specifying the ``rp`` bins for ``DDrppi``. The file should\n"
     "    contain white-space separated values  of (rpmin, rpmax)  for each\n"
     "    ``rp`` wanted. The bins do not need to be contiguous but must be in\n"
     "    increasing order (smallest bins come first). \n\n"

     "X1/Y1/Z1 : array-like, real (float/double)\n"
     "    The array of X/Y/Z positions for the first set of points.\n"
     "    Calculations are done in the precision of the supplied arrays.\n\n"

     "weights1 : array-like, real (float/double), shape (n_weights_per_particle,n_particles), optional\n"
     "    Weights for computing a weighted pair count.\n\n"

     "weight_type : str, optional\n"
     "    The type of pair weighting to apply.\n"
     "    Options: \"pair_product\", None\n"
     "    Default: None.\n\n"

     "periodic : boolean\n"
     "    Boolean flag to indicate periodic boundary conditions.\n\n"

     "X2/Y2/Z2 : array-like, real (float/double)\n"
     "    Array of XYZ positions for the second set of points. *Must* be the same\n"
     "    precision as the X1/Y1/Z1 arrays. Only required when ``autocorr==0``.\n\n"

     "weights2\n : array-like, real (float/double), shape (n_weights_per_particle,n_particles), optional\n"
     "    Weights for computing a weighted pair count.\n\n"

     "verbose : boolean (default false)\n"
     "    Boolean flag to control output of informational messages\n\n"

     "boxsize : 3-tuple of double\n"
     "    The (X,Y,Z) side lengths of the spatial domain.\n"
     "    Present to facilitate exact calculations for periodic wrapping.\n"
     "    If the boxsize in a dimension is 0., then\n"
     "    then that dimension's wrap is done based on the extent of the particle\n"
     "    distribution. If the boxsize in a dimension is -1., then periodicity\n"
     "    is disabled for that dimension.\n\n"

     "output_rpavg : boolean (default false)\n"
     "    Boolean flag to output the average ``"RP_CHAR"`` for each bin. Code will\n"
     "    run slower if you set this flag. Also, note, if you are calculating\n"
     "    in single-precision, ``"RP_CHAR"`` will suffer from numerical loss of\n"
     "    precision and can not be trusted. If you need accurate ``"RP_CHAR"``\n"
     "    values, then pass in double precision arrays for the particle positions.\n\n"

     "(xyz)bin_refine_factor: integer (default (2,2,1) typical values in [1-3]) \n"
     "    Controls the refinement on the cell sizes. Can have up to a 20% impact \n"
     "    on runtime. \n\n"

     "max_cells_per_dim: integer (default 100, typical values in [50-300]) \n"
     "    Controls the maximum number of cells per dimension. Total number of cells \n"
     "    can be up to (max_cells_per_dim)^3. Only increase if ``rmax`` is too small \n"
     "    relative to the boxsize (and increasing helps the runtime). \n\n"

     "copy_particles: boolean (default True)\n"
     "    Boolean flag to make a copy of the particle positions\n"
     "    If set to False, the particles will be re-ordered in-place\n\n"

     "enable_min_sep_opt: boolean (default true)\n"
     "    Boolean flag to allow optimizations based on min. separation between pairs \n"
     "    of cells. Here to allow for comparison studies.\n\n"

     "c_api_timer : boolean (default false)\n"
     "    Boolean flag to measure actual time spent in the C libraries. Here\n"
     "    to allow for benchmarking and scaling studies.\n\n"

     "isa : integer (default -1)\n"
     "    Controls the runtime dispatch for the instruction set to use. Possible\n"
     "    options are: [-1, AVX512F, AVX, SSE42, FALLBACK]\n\n"
     "    Setting isa to -1 will pick the fastest available instruction\n"
     "    set on the current computer. However, if you set ``isa`` to, say,\n"
     "    ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "    revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "    Setting isa to -1 will pick the fastest available instruction\n"
     "    set on the current computer. However, if you set ``isa`` to, say,\n"
     "    ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "    revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "    Unless you are benchmarking the different instruction sets, you should\n"
     "    always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "    then the integer values correspond to the ``enum`` for the instruction set\n"
     "    defined in ``utils/defs.h``.\n\n"

     "Returns\n"
     "--------\n\n"

     "A tuple (results, time) \n\n"

     "results : A python list\n"
     "    A python list containing [rpmin, rpmax, rpavg, pimax, npairs, weightavg] for each radial\n"
     "    bin specified in the ``binfile``. If ``output_rpavg`` is not set, then ``rpavg``\n"
     "    will be set to 0.0 for all bins; similarly for ``weight_avg``. ``npairs`` contains the number of pairs\n"
     "    in that bin and can be used to compute the actual wp("RP_CHAR") by\n"
     "    combining with (DR, RR) counts.\n\n"

     "time : double\n"
     "    if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "    in the API; otherwise time is set to 0.0\n\n"

     "Example\n"
     "--------\n\n"

     ">>> from Corrfunc._countpairs import countpairs_rp_pi\n"
     ">>> from Corrfunc.io import read_catalog\n"
     ">>> x,y,z = read_catalog()\n"
     ">>> autocorr=1\n"
     ">>> nthreads=2\n"
     ">>> pimax=40.0\n"
     ">>> (DDrppi, time) = countpairs_rp_pi(autocorr, nthreads, pimax, '../tests/bins',\n"
     "                                      x, y, z, X2=x, Y2=y, Z2=z,\n"
     "                                      verbose=True, output_rpavg=True)\n\n"

    },
    {"countpairs_wp"         ,(PyCFunction)(void(*)(void)) countpairs_countpairs_wp    ,METH_VARARGS | METH_KEYWORDS,
     "countpairs_wp(boxsize, pimax, nthreads, binfile, X, Y, Z, weights=None, weight_type=None, verbose=False,\n"
     "              output_rpavg=False, xbin_refine_factor=2, ybin_refine_factor=2,\n"
     "              zbin_refine_factor=1, max_cells_per_dim=100, copy_particles=True,\n"
     "              enable_min_sep_opt=True, c_api_timer=False, c_cell_timer=False, isa=-1)\n\n"

     "Function to compute the projected correlation function in a periodic\n"
     "cosmological box. Pairs which are separated by less than the ``"RP_CHAR"``\n"
     "bins (specified in ``binfile``) in the X-Y plane, and less than ``"PIMAX_CHAR"``\n"
     "in the Z-dimension are counted. *Always* uses ``PERIODIC`` boundary conditions.\n\n"

     "This module returns the actual correlation function using the natural estimator.\n"
     "Analytic randoms are used to compute wp("RP_CHAR") from the pair counts. If you\n"
     "need a different estimator, Landy-Szalay, for instance, then you should compute\n"
     "the raw pair counts with the module ``countpairs_rp_pi`` and then calculate the\n"
     "Landy-Szalay estimator for wp("RP_CHAR").\n\n"

     "If ``weights`` are provided and ``weight_type`` is \"pair_product\", then a weighted correlation function is returned.\n\n"
     "Note that pairs are double-counted. And if ``rpmin`` is set to\n"
     "0.0, then all the self-pairs (i'th particle with itself) are\n"
     "added to the first bin => minimum number of pairs in the first bin\n"
     "is the total number of particles. Also note that the python wrapper\n"
     "for this extension: `Corrfunc.theory.wp` is more user-friendly.\n"
     UNICODE_WARNING
     "\n"
     "Parameters\n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n\n"

     "boxsize: double\n"
     "    A double-precision value for the boxsize of the simulation\n"
     "    in same units as the particle positions and the ``rp`` bins.\n\n"

     "pimax: double\n"
     "    A double-precision value for the maximum separation along\n"
     "    the Z-dimension. Note that only pairs with ``0 <= dz < pimax``\n"
     "    are counted (no equality).\n"
     "    Distances along the Z direction ("PI_CHAR") are binned with unit\n"
     "    depth. For instance, if ``pimax=40``, then 40 bins will be created\n"
     "    along the ``"PI_CHAR"`` direction.\n\n"

     "nthreads: integer\n"
     "    Number of threads to use.\n\n"

     "binfile : string\n"
     "    Filename specifying the ``rp`` bins for ``wp``. The file should\n"
     "    contain white-space separated values  of (rpmin, rpmax)  for each\n"
     "    ``rp`` wanted. The bins do not need to be contiguous but must be in\n"
     "    increasing order (smallest bins come first). \n\n"

     "X/Y/Z : array-like, real (float/double)\n"
     "    The array of X/Y/Z positions for the first set of points.\n"
     "    Calculations are done in the precision of the supplied arrays.\n\n"

     "weights : array-like, real (float/double), shape (n_particles,) or (n_weights_per_particle,n_particles), optional\n"
     "    Weights for computing a weighted correlation function.\n\n"

     "weight_type : str, optional\n"
     "    The type of pair weighting to apply.\n"
     "    Options: \"pair_product\", None\n"
     "    Default: None.\n\n"

     "verbose : boolean (default false)\n"
     "    Boolean flag to control output of informational messages\n\n"

     "output_rpavg : boolean (default false)\n"
     "    Boolean flag to output the average ``"RP_CHAR"`` for each bin. Code will\n"
     "    run slower if you set this flag. Also, note, if you are calculating\n"
     "    in single-precision, ``"RP_CHAR"`` will suffer from numerical loss of\n"
     "    precision and can not be trusted. If you need accurate ``"RP_CHAR"``\n"
     "    values, then pass in double precision arrays for the particle positions.\n\n"

     "(xyz)bin_refine_factor: integer (default (2,2,1) typical values in [1-3]) \n"
     "    Controls the refinement on the cell sizes. Can have up to a 20% impact \n"
     "    on runtime. \n\n"

     "max_cells_per_dim: integer (default 100, typical values in [50-300]) \n"
     "    Controls the maximum number of cells per dimension. Total number of cells \n"
     "    can be up to (max_cells_per_dim)^3. Only increase if ``rmax`` is too small \n"
     "    relative to the boxsize (and increasing helps the runtime). \n\n"

     "copy_particles: boolean (default True)\n"
     "    Boolean flag to make a copy of the particle positions\n"
     "    If set to False, the particles will be re-ordered in-place\n\n"

     "enable_min_sep_opt: boolean (default true)\n"
     "    Boolean flag to allow optimizations based on min. separation between pairs \n"
     "    of cells. Here to allow for comparison studies.\n\n"

     "c_api_timer : boolean (default false)\n"
     "    Boolean flag to measure actual time spent in the C libraries. Here\n"
     "    to allow for benchmarking and scaling studies.\n\n"

     "c_cell_timer : boolean (default false)\n"
     "    Boolean flag to measure actual time spent **per cell-pair** within the C libraries.\n"
     "    A very detailed timer that stores information about the number of particles in\n"
     "    each cell, the thread id that processed that cell-pair and the amount of time in\n"
     "    nano-seconds taken to process that cell pair. This timer can be used to study\n"
     "    the instruction set efficiency, and load-balancing of the code\n\n"

     "isa : integer (default -1)\n"
     "    Controls the runtime dispatch for the instruction set to use. Possible\n"
     "    options are: [-1, AVX512F, AVX, SSE42, FALLBACK]\n\n"
     "    Setting isa to -1 will pick the fastest available instruction\n"
     "    set on the current computer. However, if you set ``isa`` to, say,\n"
     "    ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "    revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "    Unless you are benchmarking the different instruction sets, you should\n"
     "    always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "    then the integer values correspond to the ``enum`` for the instruction set\n"
     "    defined in ``utils/defs.h``.\n\n"

     "Returns\n"
     "--------\n\n"

     "A tuple of (results, time, per_cell_time) \n\n"

     "results : A python list\n"
     "    A python list containing [rpmin, rpmax, rpavg, wp, npairs, weight_avg] for each radial\n"
     "    bin specified in the ``binfile``. If ``output_rpavg`` is not set then\n"
     "    ``rpavg`` will be set to 0.0 for all bins; similarly for ``weight_avg``. ``wp`` contains the projected\n"
     "    correlation function while ``npairs`` contains the number of unique pairs\n"
     "    in that bin.  If weight are used, then ``wp`` is weighted, while ``npairs`` is not.\n\n"

     "time : double\n"
     "    if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "    in the API; otherwise time is set to 0.0\n\n"

     "per_cell_time : python list of doubles\n"
     "    if ``c_cell_timer`` is set, then a Python list is returned containing\n"
     "    detailed stats about each cell-pair visited during pair-counting, viz., number of\n"
     "    particles in each of the cells in the pair, 1-D cell-indices for each cell in the pair,\n"
     "    time (in nano-seconds) to process the pair and the thread-id for the thread that \n"
     "    processed that cell-pair.\n\n"

     "Example\n"
     "--------\n\n"

     ">>> from _countpairs import countpairs_wp\n"
     ">>> from Corrfunc.io import read_catalog\n"
     ">>> x,y,z = read_catalog()\n"
     ">>> nthreads=2\n"
     ">>> pimax=40.0\n"
     ">>> boxsize = 420.0\n"
     ">>> (wp, time) = countpairs_wp(boxsize, nthreads, pimax, '../tests/bins',\n"
     "                               x, y, z, verbose=True, output_rpavg=True)\n\n"
    },
    {"countpairs_xi"         ,(PyCFunction)(void(*)(void)) countpairs_countpairs_xi    ,METH_VARARGS | METH_KEYWORDS,
     "countpairs_xi(boxsize, nthreads, binfile, X, Y, Z, weights=None, weight_type=None, verbose=False,\n"
     "              output_ravg=False, xbin_refine_factor=2, ybin_refine_factor=2,\n"
     "              zbin_refine_factor=1, max_cells_per_dim=100, copy_particles=True,\n"
     "              enable_min_sep_opt=True, c_api_timer=False, isa=-1)\n\n"

     "Function to compute the correlation function in a periodic\n"
     "cosmological box. Pairs which are separated by less than the ``r``\n"
     "bins (specified in ``binfile``). *Always* uses ``PERIODIC`` boundary conditions.\n\n"

     "This module returns the actual correlation function using the natural estimator.\n"
     "Analytic randoms are used to compute "XI_CHAR"(r) from the pair counts. If you\n"
     "need a different estimator, Landy-Szalay, for instance, then you should compute\n"
     "the raw pair counts with the module ``countpairs`` and then calculate the\n"
     "Landy-Szalay estimator for "XI_CHAR"(r).\n\n"

     "If ``weights`` are provided and ``weight_type`` is \"pair_product\", then a weighted correlation function is returned.\n\n"
     "Note that pairs are double-counted. And if ``rmin`` is set to\n"
     "0.0, then all the self-pairs (i'th particle with itself) are\n"
     "added to the first bin => minimum number of pairs in the first bin\n"
     "is the total number of particles. Also note that the python wrapper\n"
     "for this extension: `Corrfunc.theory.xi` is more user-friendly.\n"
     UNICODE_WARNING"\n"

     "Parameters\n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n\n"

     "boxsize: double\n"
     "    A double-precision value for the boxsize of the simulation\n"
     "    in same units as the particle positions and the ``r`` bins.\n\n"

     "nthreads: integer\n"
     "    Number of threads to use.\n\n"

     "binfile : string\n"
     "    Filename specifying the ``r`` bins for ``xi``. The file should\n"
     "    contain white-space separated values  of (rmin, rmax)  for each\n"
     "    ``r`` wanted. The bins do not need to be contiguous but must be in\n"
     "    increasing order (smallest bins come first). \n\n"

     "X1/Y1/Z1 : array-like, real (float/double)\n"
     "    The array of X/Y/Z positions for the first set of points.\n"
     "    Calculations are done in the precision of the supplied arrays.\n\n"

     "weights : array-like, real (float/double), shape (n_particles,) or (n_weights_per_particle,n_particles), optional\n"
     "    Weights for computing a weighted correlation function.\n\n"

     "weight_type : str, optional\n"
     "    The type of pair weighting to apply.\n"
     "    Options: \"pair_product\", None\n"
     "    Default: None.\n\n"

     "verbose : boolean (default false)\n"
     "    Boolean flag to control output of informational messages\n\n"

     "output_ravg : boolean (default false)\n"
     "    Boolean flag to output the average ``r`` for each bin. Code will\n"
     "    run slower if you set this flag. Also, note, if you are calculating\n"
     "    in single-precision, ``r`` will suffer from numerical loss of\n"
     "    precision and can not be trusted. If you need accurate ``r``\n"
     "    values, then pass in double precision arrays for the particle positions.\n\n"

     "(xyz)bin_refine_factor: integer (default (2,2,1) typical values in [1-3]) \n"
     "    Controls the refinement on the cell sizes. Can have up to a 20% impact \n"
     "    on runtime. \n\n"

     "max_cells_per_dim: integer (default 100, typical values in [50-300]) \n"
     "    Controls the maximum number of cells per dimension. Total number of cells \n"
     "    can be up to (max_cells_per_dim)^3. Only increase if ``rmax`` is too small \n"
     "    relative to the boxsize (and increasing helps the runtime). \n\n"

     "copy_particles: boolean (default True)\n"
     "    Boolean flag to make a copy of the particle positions\n"
     "    If set to False, the particles will be re-ordered in-place\n\n"

     "enable_min_sep_opt: boolean (default true)\n"
     "    Boolean flag to allow optimizations based on min. separation between pairs \n"
     "    of cells. Here to allow for comparison studies.\n\n"

     "c_api_timer : boolean (default false)\n"
     "    Boolean flag to measure actual time spent in the C libraries. Here\n"
     "    to allow for benchmarking and scaling studies.\n\n"

     "isa : integer (default -1)\n"
     "    Controls the runtime dispatch for the instruction set to use. Possible\n"
     "    options are: [-1, AVX512F, AVX, SSE42, FALLBACK]\n\n"
     "    Setting isa to -1 will pick the fastest available instruction\n"
     "    set on the current computer. However, if you set ``isa`` to, say,\n"
     "    ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "    revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "    Unless you are benchmarking the different instruction sets, you should\n"
     "    always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "    then the integer values correspond to the ``enum`` for the instruction set\n"
     "    defined in ``utils/defs.h``.\n\n"

     "Returns\n"
     "--------\n\n"

     "A tuple (results, time) \n\n"

     "results : A python list\n"
     "    A python list containing [rmin, rmax, ravg, weightavg, xi, npairs] for each radial\n"
     "    bin specified in the ``binfile``. If ``output_ravg`` is not set then\n"
     "    ``ravg`` will be set to 0.0 for all bins; similarly for ``weightavg``. ``xi`` contains the\n"
     "    correlation function while ``npairs`` contains the number of unique pairs\n"
     "    in that bin.  If weights are used, then ``xi`` is weighted, while ``npairs`` is not.\n\n"

     "time : double\n"
     "    if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "    in the API; otherwise time is set to 0.0\n\n"

     "Example\n"
     "--------\n\n"

     ">>> from _countpairs import countpairs_xi\n"
     ">>> from Corrfunc.io import read_catalog\n"
     ">>> x,y,z = read_catalog()\n"
     ">>> nthreads=2\n"
     ">>> boxsize = 420.0\n"
     ">>> (xi, time) = countpairs_xi(boxsize, nthreads, '../tests/bins',\n"
     "                               x, y, z, verbose=True, output_ravg=True)\n"
     "\n"
    },
    {"countpairs_s_mu"      ,(PyCFunction)(void(*)(void)) countpairs_countpairs_s_mu ,METH_VARARGS | METH_KEYWORDS,
     "countpairs_s_mu(autocorr, nthreads, binfile, mu_max, nmu_bins, X1, Y1, Z1, weights1=None, weight_type=None,\n"
     "                periodic=True, X2=None, Y2=None, Z2=None, weights2=None, verbose=False,\n"
     "                boxsize=0.0, output_savg=False, fast_divide_and_NR_steps=0,\n"
     "                xbin_refine_factor=2, ybin_refine_factor=2, zbin_refine_factor=1,\n"
     "                max_cells_per_dim=100, copy_particles=True,\n"
     "                enable_min_sep_opt=True, c_api_timer=False, isa=-1)\n"
     "\n"
     "Calculate the 2-D pair-counts corresponding to the real-space correlation\n"
     "function, "XI_CHAR"(s, "MU_CHAR"). Pairs which are separated\n"
     "by less than the ``s`` bins (specified in ``binfile``) in the X-Y plane, and\n"
     "less than ``s*mu_max`` in the Z-dimension are counted.\n\n"

     "Note, that this module only returns pair counts and not the actual\n"
     "correlation function "XI_CHAR"(s, "MU_CHAR"). \n"
     "Also note that the python wrapper for this extension: `Corrfunc.theory.DDsmu`\n"
     "is more user-friendly.\n"
     UNICODE_WARNING"\n"

     "Parameters\n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n\n"

     "autocorr: boolean, required\n"
     "    Boolean flag for auto/cross-correlation. If autocorr is set to 1,\n"
     "    are not used (but must still be passed, perhaps again as X1/Y1/Z1).\n\n"

     "nthreads: integer\n"
     "    The number of OpenMP threads to use. Has no effect if OpenMP was not\n"
     "    enabled during library compilation.\n\n"

     "binfile : string\n"
     "    Filename specifying the ``s`` bins for ``DDsmu``. The file should\n"
     "    contain white-space separated values  of (smin, smax)  for each\n"
     "    ``s`` wanted. The bins must be contiguous and in\n"
     "    increasing order (smallest bins come first). \n\n"

     "mu_max: double. Must be in range (0.0, 1.0]\n"
     "    A double-precision value for the maximum cosine of the angular separation from\n"
     "    the line of sight (LOS). Here, LOS is taken to be along the Z direction.\n"
     "    Note that only pairs with ``0 <= cos("THETA_CHAR"_LOS) < mu_max``\n"
     "    are counted (no equality).\n\n"

     "nmu_bins: Integer. Must be at least 1\n"
     "    Number of bins for ``mu``\n\n"

     "X1/Y1/Z1 : array-like, real (float/double)\n"
     "    The array of X/Y/Z positions for the first set of points.\n"
     "    Calculations are done in the precision of the supplied arrays.\n\n"

     "weights1 : array-like, real (float/double), shape (n_weights_per_particle,n_particles), optional\n"
     "    Weights for computing a weighted pair count.\n\n"

     "weight_type : str, optional\n"
     "    The type of pair weighting to apply.\n"
     "    Options: \"pair_product\", None\n"
     "    Default: None.\n\n"

     "periodic : boolean\n"
     "    Boolean flag to indicate periodic boundary conditions.\n\n"

     "X2/Y2/Z2 : array-like, real (float/double)\n"
     "    Array of XYZ positions for the second set of points. *Must* be the same\n"
     "    precision as the X1/Y1/Z1 arrays. Only required when ``autocorr==0``.\n\n"

     "weights2\n : array-like, real (float/double), shape (n_weights_per_particle,n_particles), optional\n"
     "    Weights for computing a weighted pair count.\n\n"

     "verbose : boolean (default false)\n"
     "    Boolean flag to control output of informational messages\n\n"

     "boxsize : 3-tuple of double\n"
     "    The (X,Y,Z) side lengths of the spatial domain.\n"
     "    Present to facilitate exact calculations for periodic wrapping.\n"
     "    If the boxsize in a dimension is 0., then\n"
     "    then that dimension's wrap is done based on the extent of the particle\n"
     "    distribution. If the boxsize in a dimension is -1., then periodicity\n"
     "    is disabled for that dimension.\n\n"

     "output_savg : boolean (default false)\n"
     "    Boolean flag to output the average ``s`` for each bin. Code will\n"
     "    run slower if you set this flag. Also, note, if you are calculating\n"
     "    in single-precision, ``s`` will suffer from numerical loss of\n"
     "    precision and can not be trusted. If you need accurate ``s``\n"
     "    values, then pass in double precision arrays for the particle positions.\n\n"

     "fast_divide_and_NR_steps: integer (default 0)\n"
     "    Replaces the division in ``AVX512F`` and ``AVX`` kernels with an\n"
     "    approximate reciprocal, followed by ``fast_divide_and_NR_steps`` "
     "    Newton-Raphson step. Can improve \n"
     "    runtime by ~15-20%. Value of 0 keeps the standard division.\n\n"

     "(xyz)bin_refine_factor: integer (default (2,2,1) typical values in [1-3]) \n"
     "    Controls the refinement on the cell sizes. Can have up to a 20% impact \n"
     "    on runtime. \n\n"

     "max_cells_per_dim: integer (default 100, typical values in [50-300]) \n"
     "    Controls the maximum number of cells per dimension. Total number of cells \n"
     "    can be up to (max_cells_per_dim)^3. Only increase if ``rmax`` is too small \n"
     "    relative to the boxsize (and increasing helps the runtime). \n\n"

     "copy_particles: boolean (default True)\n"
     "    Boolean flag to make a copy of the particle positions\n"
     "    If set to False, the particles will be re-ordered in-place\n\n"

     "enable_min_sep_opt: boolean (default true)\n"
     "    Boolean flag to allow optimizations based on min. separation between pairs \n"
     "    of cells. Here to allow for comparison studies.\n\n"

     "c_api_timer : boolean (default false)\n"
     "    Boolean flag to measure actual time spent in the C libraries. Here\n"
     "    to allow for benchmarking and scaling studies.\n\n"

     "isa : integer (default -1)\n"
     "    Controls the runtime dispatch for the instruction set to use. Possible\n"
     "    options are: [-1, AVX512F, AVX, SSE42, FALLBACK]\n\n"
     "    Setting isa to -1 will pick the fastest available instruction\n"
     "    set on the current computer. However, if you set ``isa`` to, say,\n"
     "    ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "    revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "    Unless you are benchmarking the different instruction sets, you should\n"
     "    always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "    then the integer values correspond to the ``enum`` for the instruction set\n"
     "    defined in ``utils/defs.h``.\n\n"

     "Returns\n"
     "--------\n\n"

     "A tuple (results, time) \n\n"

     "results : A python list\n"
     "    A python list containing ``nmu_bins`` of [smin, smax, savg, mu_max, npairs, weightavg]\n"
     "    for each spatial bin specified in the ``binfile``. There will be a total of ``nmu_bins``\n"
     "    ranging from [0, ``mu_max``) *per* spatial bin. If ``output_savg`` is not set, then ``savg``\n"
     "    will be set to 0.0 for all bins; similarly for ``weight_avg``. ``npairs`` \n"
     "    contains the number of pairs in that bin.\n\n"

     "time : double\n"
     "    if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "    in the API; otherwise time is set to 0.0\n\n"

     "Example\n"
     "--------\n\n"

     ">>> from Corrfunc._countpairs import countpairs_s_mu\n"
     ">>> from Corrfunc.io import read_catalog\n"
     ">>> x,y,z = read_catalog()\n"
     ">>> autocorr=1\n"
     ">>> nthreads=2\n"
     ">>> mu_max=1.0\n"
     ">>> nmu_bins=40\n"
     ">>> (DDsmu, time) = countpairs_s_mu(autocorr, nthreads, '../tests/bins', mu_max, nmu_bins, \n"
     "                                    x, y, z, X2=x, Y2=y, Z2=z,\n"
     "                                    verbose=True, output_savg=True)\n\n"
    },
    {"countspheres_vpf"      ,(PyCFunction)(void(*)(void)) countpairs_countspheres_vpf ,METH_VARARGS | METH_KEYWORDS,
     "countspheres_vpf(rmax, nbins, nspheres, numpN, seed,\n"
     "                 X, Y, Z, verbose=False, periodic=True,\n"
     "                 boxsize=0.0, xbin_refine_factor=1, ybin_refine_factor=1,\n"
     "                 zbin_refine_factor=1, max_cells_per_dim=100, copy_particles=True,\n"
     "                 c_api_timer=False, isa=-1)\n\n"

     "Calculates the fraction of random spheres that contain exactly *N* points, pN(r).\n\n"
     UNICODE_WARNING"\n"

     "Parameters\n"
     "-----------\n\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n\n"

     "rmax: double\n"
     "    Maximum radius of the sphere to place on the particles\n\n"

     "nbins: integer\n"
     "    Number of bins in the counts-in-cells. Radius of first shell\n"
     "    is rmax/nbins\n\n"

     "nspheres: integer (>= 0)\n"
     "    Number of random spheres to place within the particle distribution.\n"
     "    For a small number of spheres, the error is larger in the measured\n"
     "    pN's.\n\n"

     "numpN: integer (>= 1)\n"
     "    Governs how many unique pN's are to returned. If ``numpN` is set to 1,\n"
     "    then only the vpf (p0) is returned. For ``numpN=2``, p0 and p1 are\n"
     "    returned.\n\n"

     "    More explicitly, the columns in the results look like the following:\n"
     "      numpN = 1 -> p0\n"
     "      numpN = 2 -> p0 p1\n"
     "      numpN = 3 -> p0 p1 p2\n"
     "      and so on...(note that p0 is the vpf).\n\n"

     "seed: unsigned integer\n"
     "    Random number seed for the underlying random number generator. Used\n"
     "    to draw centers of the spheres.\n\n"

     "X/Y/Z: arraytype, real (float/double)\n"
     "    Particle positions in the 3 axes. Must be within [0, boxsize]\n"
     "    and specified in the same units as ``rp_bins`` and boxsize. All\n"
     "    3 arrays must be of the same floating-point type.\n"
     "\n"
     "    Calculations will be done in the same precision as these arrays,\n"
     "    i.e., calculations will be in floating point if XYZ are single\n"
     "    precision arrays (C float type); or in double-precision if XYZ\n"
     "    are double precision arrays (C double type).\n\n"

     "verbose: boolean (default false)\n"
     "    Boolean flag to control output of informational messages\n\n"

     "periodic: boolean\n"
     "    Boolean flag to indicate periodic boundary conditions.\n\n"

     "boxsize : 3-tuple of double\n"
     "    The (X,Y,Z) side lengths of the spatial domain.\n"
     "    Present to facilitate exact calculations for periodic wrapping.\n"
     "    If the boxsize in a dimension is 0., then\n"
     "    then that dimension's wrap is done based on the extent of the particle\n"
     "    distribution. If the boxsize in a dimension is -1., then periodicity\n"
     "    is disabled for that dimension.\n\n"

     "(xyz)bin_refine_factor: integer (default (1,1,1) typical values in [1-3]) \n"
     "    Controls the refinement on the cell sizes. Can have up to a 20% impact \n"
     "    on runtime. Note that the default values are different from the \n"
     "    correlation function routines. \n\n"

     "max_cells_per_dim: integer (default 100, typical values in [50-300]) \n"
     "    Controls the maximum number of cells per dimension. Total number of cells \n"
     "    can be up to (max_cells_per_dim)^3. Only increase if ``rmax`` is too small \n"
     "    relative to the boxsize (and increasing helps the runtime). \n\n"

     "copy_particles: boolean (default True)\n"
     "    Boolean flag to make a copy of the particle positions\n"
     "    If set to False, the particles will be re-ordered in-place\n\n"

     "c_api_timer : boolean (default false)\n"
     "    Boolean flag to measure actual time spent in the C libraries. Here\n"
     "    to allow for benchmarking and scaling studies.\n\n"

     "isa : integer (default -1)\n"
     "    Controls the runtime dispatch for the instruction set to use. Possible\n"
     "    options are: [-1, AVX512F, AVX, SSE42, FALLBACK]\n\n"
     "    Setting isa to -1 will pick the fastest available instruction\n"
     "    set on the current computer. However, if you set ``isa`` to, say,\n"
     "    ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "    revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "    Unless you are benchmarking the different instruction sets, you should\n"
     "    always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "    then the integer values correspond to the ``enum`` for the instruction set\n"
     "    defined in ``utils/defs.h``.\n\n"

     "Returns\n"
     "--------\n\n"

     "A tuple (results, time) \n\n"

     "results : a Python list \n"
     "    The list contains [rmax, p0, p1,..., p(num_pN-1)] for each radial bin.\n\n"

     "time : double\n"
     "    if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "    in the API; otherwise time is set to 0.0\n\n"

     "Example\n"
     "--------\n\n"

     ">>> from _countpairs import countspheres_vpf\n"
     ">>> from Corrfunc.io import read_catalog\n"
     ">>> rmax = 10.0\n"
     ">>> nbins = 10\n"
     ">>> nspheres = 10000\n"
     ">>> numpN = 8\n"
     ">>> seed = -1\n"
     ">>> boxsize = 420.0\n"
     ">>> X, Y, Z = read_catalog()\n"
     ">>> results, api_time = countspheres_vpf(rmax, nbins, nspheres, numpN, seed,\n"
     "                                         X, Y, Z,\n"
     "                                         verbose=True,\n"
     "                                         c_api_timer=True,\n"
     "                                         boxsize=boxsize,\n"
     "                                         periodic=True)\n\n"
    },
    {NULL, NULL, 0, NULL}
};

static PyObject *countpairs_error_out(PyObject *module, const char *msg)
{
#if PY_MAJOR_VERSION < 3
    (void) module;//to avoid unused warning with python2
#endif

    struct module_state *st = GETSTATE(module);
    PyErr_SetString(st->error, msg);
    PyErr_Print();
    Py_RETURN_NONE;
}


#if PY_MAJOR_VERSION >= 3
static int _countpairs_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _countpairs_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_countpairs",
    module_docstring,
    sizeof(struct module_state),
    module_methods,
    NULL,
    _countpairs_traverse,
    _countpairs_clear,
    NULL
};


PyObject *PyInit__countpairs(void)
#else
//Python 2
PyMODINIT_FUNC init_countpairs(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule3("_countpairs", module_methods, module_docstring);
#endif

    if (module == NULL) {
        INITERROR;
    }

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("_countpairs.error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    /* Load `numpy` functionality. */
    import_array();

    highest_isa = get_max_usable_isa();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif

}
// weights1_obj may be NULL, in which case it is ignored.
// If it is not NULL, it will be checked alongside the positions
static int64_t check_dims_and_datatype(PyObject *module, PyArrayObject *x1_obj, PyArrayObject *y1_obj, PyArrayObject *z1_obj, size_t *element_size)
{
    char msg[1024];

    /* All the position arrays should be 1-D*/
    const int nxdims = PyArray_NDIM(x1_obj);
    const int nydims = PyArray_NDIM(y1_obj);
    const int nzdims = PyArray_NDIM(z1_obj);

    if(nxdims != 1 || nydims != 1 || nzdims != 1) {
        snprintf(msg, 1024, "ERROR: Expected 1-D numpy arrays.\nFound (nxdims, nydims, nzdims) = (%d, %d, %d) instead",
                 nxdims, nydims, nzdims);
        countpairs_error_out(module, msg);
        return -1;
    }

    /* All the arrays should be floating point (only float32 and float64 are allowed) */
    const int x_type = PyArray_TYPE(x1_obj);
    const int y_type = PyArray_TYPE(y1_obj);
    const int z_type = PyArray_TYPE(z1_obj);
    if( ! ((x_type == NPY_FLOAT || x_type == NPY_DOUBLE) &&
           (y_type == NPY_FLOAT || y_type == NPY_DOUBLE) &&
           (z_type == NPY_FLOAT || z_type == NPY_DOUBLE))
        ) {
        PyArray_Descr *x_descr = PyArray_DescrFromType(x_type);
        PyArray_Descr *y_descr = PyArray_DescrFromType(y_type);
        PyArray_Descr *z_descr = PyArray_DescrFromType(z_type);
        if(x_descr == NULL || y_descr == NULL || z_descr == NULL) {
            /* Generating the dtype descriptor failed somehow. At least provide some information */
            snprintf(msg, 1024, "TypeError: Expected floating point arrays (allowed types = %d or %d). Instead found type-nums (%d, %d, %d)\n",
                     NPY_FLOAT, NPY_DOUBLE, x_type, y_type, z_type);
        } else {
            snprintf(msg, 1024, "TypeError: Expected floating point arrays (allowed types = %d or %d). Instead found type-nums (%d, %d, %d) "
                     "with type-names = (%s, %s, %s)\n",
                     NPY_FLOAT, NPY_DOUBLE, x_type, y_type, z_type, x_descr->typeobj->tp_name, y_descr->typeobj->tp_name, z_descr->typeobj->tp_name);
        }
        Py_XDECREF(x_descr);Py_XDECREF(y_descr);Py_XDECREF(z_descr);
        countpairs_error_out(module, msg);
        return -1;
    }

    // Current version of the code only supports weights of the same dtype as positions
    if( x_type != y_type || y_type != z_type) {
        PyArray_Descr *x_descr = PyArray_DescrFromType(x_type);
        PyArray_Descr *y_descr = PyArray_DescrFromType(y_type);
        PyArray_Descr *z_descr = PyArray_DescrFromType(z_type);
        if(x_descr == NULL || y_descr == NULL || z_descr == NULL) {
            /* Generating the dtype descriptor failed somehow. At least provide some information */
            snprintf(msg, 1024, "TypeError: Expected *ALL* 3 floating point arrays to be the same type (allowed types = %d or %d). Instead found type-nums (%d, %d, %d)\n",
                     NPY_FLOAT, NPY_DOUBLE, x_type, y_type, z_type);
        } else {
            snprintf(msg, 1024, "TypeError: Expected *ALL* 3 floating point arrays to be the same type (allowed types = %d or %d). Instead found type-nums (%d, %d, %d) "
                     "with type-names = (%s, %s, %s)\n",
                     NPY_FLOAT, NPY_DOUBLE, x_type, y_type, z_type, x_descr->typeobj->tp_name, y_descr->typeobj->tp_name, z_descr->typeobj->tp_name);
        }
        Py_XDECREF(x_descr);Py_XDECREF(y_descr);Py_XDECREF(z_descr);
        countpairs_error_out(module, msg);
        return -1;
    }

    /* Check if the number of elements in the 3 Python arrays are identical */
    const int64_t nx1 = (int64_t)PyArray_SIZE(x1_obj);
    const int64_t ny1 = (int64_t)PyArray_SIZE(y1_obj);
    const int64_t nz1 = (int64_t)PyArray_SIZE(z1_obj);

    if(nx1 != ny1 || ny1 != nz1) {
      snprintf(msg, 1024, "ERROR: Expected arrays to have the same number of elements in all 3-dimensions.\nFound (nx, ny, nz) = (%"PRId64", %"PRId64", %"PRId64") instead",
               nx1, ny1, nz1);
      countpairs_error_out(module, msg);
      return -1;
    }

    /* Return the size of each element of the data object */
    if(x_type == NPY_FLOAT) {
      *element_size = sizeof(float);
    } else {
      *element_size = sizeof(double);
    }

    return nx1;
}


static int print_kwlist_into_msg(char *msg, const size_t totsize, size_t len, char *kwlist[], const size_t nitems)
{
    for(size_t i=0;i<nitems;i++) {

        if(len+strlen(kwlist[i]) >= totsize-2) {
            return EXIT_FAILURE;
        }

        memcpy(msg+len, kwlist[i], strlen(kwlist[i]));
        len += strlen(kwlist[i]);
        msg[len] = ',';
        msg[len+1] = ' ';
        len += 2;
    }

    msg[len]='\0';
    return EXIT_SUCCESS;
}


static int check_weights(PyObject *module, PyObject *weights_obj, weight_struct *weight_st, const weight_method_t method, const int64_t nx, size_t element_size)
{
    int status = EXIT_SUCCESS;
    char msg[1024];
    if (!PySequence_Check(weights_obj)) {
      snprintf(msg, 1024, "Please input tuple/list of weights");
      goto except;
    }
    PyObject *iter_arrays = NULL, *array_obj = NULL;
    PyArrayObject *array = NULL;
    iter_arrays = PyObject_GetIter(weights_obj); // raises error if NULL
    if (iter_arrays == NULL) goto except;
    int64_t w = 0;
    weight_type_t itemtype[MAX_NUM_WEIGHTS];
    const int requirements = NPY_ARRAY_IN_ARRAY;
    while ((array_obj = PyIter_Next(iter_arrays))) {
        array = (PyArrayObject *) PyArray_FromArray((PyArrayObject *) array_obj, NOTYPE_DESCR, requirements);
        if (array == NULL) {
            snprintf(msg, 1024, "TypeError: Could not convert input weights to arrays. Are you passing numpy arrays?\n");
            goto except_iter;
        }
        /* The weights array must be 2-D of shape (n_weights, n_particles) */
        const int ndims = PyArray_NDIM(array);
        if (ndims != 1) {
            snprintf(msg, 1024, "ERROR: Expected 1-D numpy arrays.\nFound ndims = %d instead.\n", ndims);
            goto except_iter;
        }
        const int64_t nx_weights = (int64_t) PyArray_SIZE((PyArrayObject *) array);
        if (nx_weights != nx) {
            snprintf(msg, 1024, "ERROR: Expected weight arrays to have the same number of elements as input coordinates.\nFound nx = %"PRId64" instead.\n", nx_weights);
            goto except_iter;
        }
        const int array_type = PyArray_TYPE(array);
        switch (array_type) {
            case NPY_FLOAT:
                itemtype[w] = FLOAT_TYPE;
                if (element_size != sizeof(float)) {
                    snprintf(msg, 1024, "ERROR: Input coordinates are float32 but provided weights are float64. Please use the same size.\n");
                    goto except_iter;
                }
                break;
            case NPY_DOUBLE:
                itemtype[w] = FLOAT_TYPE;
                if (element_size != sizeof(double)) {
                    snprintf(msg, 1024, "ERROR: Input coordinates are float64 but provided weights are float32. Please use the same size.\n");
                    goto except_iter;
                }
                break;
            case NPY_INT32:
                itemtype[w] = INT_TYPE;
                if (element_size != sizeof(int32_t)) {
                    snprintf(msg, 1024, "ERROR: Input coordinates are float64 but provided weights are int32. Please use the same size.\n");
                    goto except_iter;
                }
                break;
            case NPY_INT64:
                itemtype[w] = INT_TYPE;
                if (element_size != sizeof(int64_t)) {
                    snprintf(msg, 1024, "ERROR: Input coordinates are float64 but provided weights are int32. Please use the same size.\n");
                    goto except_iter;
                }
                break;
            default:
                snprintf(msg, 1024, "TypeError: Expected integer or floating arrays for weights. Instead found type-num %d.\n", array_type);
                goto except_iter;
        }

        if (array_type == NPY_FLOAT || array_type == NPY_DOUBLE) {
            itemtype[w] = FLOAT_TYPE;
        }
        else if (array_type == NPY_INT32 || array_type == NPY_INT64) {
            itemtype[w] = INT_TYPE;
        }
        else {
            snprintf(msg, 1024, "ERROR: Unknown weight array type.\n");
            goto except_iter;
        }
        weight_st->weights[w] = (void *) PyArray_DATA(array);
        w++;
        goto finally_iter;
except_iter:
        Py_XDECREF(array_obj);
        Py_XDECREF(array);
        goto except;
finally_iter:
        Py_XDECREF(array_obj);
        Py_XDECREF(array);
    }
    status = set_weight_struct(weight_st, method, itemtype, w);
    goto finally;
except:
    countpairs_error_out(module, msg);
    return EXIT_FAILURE;
finally:
    Py_XDECREF(iter_arrays);
    return status;
}


static int check_pair_weight(PyObject *module, pair_weight_struct *pair_weight_st, PyObject *sep_obj, PyObject *weight_obj, size_t element_size, PyObject *attrs_pair_weight_obj)
{
    int status = EXIT_SUCCESS;
    char msg[1024];
    pair_weight_st->noffset = 1;
    pair_weight_st->default_value = 0.;
    PyArrayObject *sep = NULL, *weight = NULL;
    if (attrs_pair_weight_obj != NULL) {
        if (!PyDict_Check(attrs_pair_weight_obj)) {
            snprintf(msg, 1024, "Please input dict of name: value for attrs_pair_weights");
            goto except;
        }
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(attrs_pair_weight_obj, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "noffset") == 0) {
                pair_weight_st->noffset = PyLong_AsLong(value);
            }
            else if (PyUnicode_CompareWithASCIIString(key, "default_value") == 0) {
                pair_weight_st->default_value = PyFloat_AsDouble(value);
            }
            else if (PyUnicode_CompareWithASCIIString(key, "correction") == 0) {
                const int requirements = NPY_ARRAY_IN_ARRAY;
                PyArrayObject *array = (PyArrayObject *) PyArray_FromArray((PyArrayObject *) value, NOTYPE_DESCR, requirements);
                if (array == NULL) {
                    snprintf(msg, 1024, "TypeError: Could not convert 'correction' of attrs_pair_weights to array. Are you passing a numpy array?\n");
                    Py_XDECREF(array);
                    goto except;
                }
                pair_weight_st->correction_bits = (void *) PyArray_DATA(array);
                pair_weight_st->num_bits = (int8_t) sqrt((double) PyArray_SIZE((PyArrayObject *) array));
                Py_XDECREF(array);
            }
            else {
                snprintf(msg, 1024, "ERROR: Found unknown key in attrs_pair_weights\n");
                goto except;
            }
        }
    }
    if (weight_obj == NULL) return status;
    const int requirements = NPY_ARRAY_IN_ARRAY;
    sep = (PyArrayObject *) PyArray_FromArray((PyArrayObject *) sep_obj, NOTYPE_DESCR, requirements);
    weight = (PyArrayObject *) PyArray_FromArray((PyArrayObject *) weight_obj, NOTYPE_DESCR, requirements);
    PyArrayObject *arrays[2] = {sep, weight};
    for (int ii=0; ii<2; ii++) {
        PyArrayObject *array = arrays[ii];
        if (array == NULL) {
            snprintf(msg, 1024, "TypeError: Could not convert input pair weight to array. Are you passing numpy array?\n");
            goto except;
        }
        const int ndims = PyArray_NDIM(array);
        if (ndims != 1) {
            snprintf(msg, 1024, "ERROR: Expected 1-D numpy arrays.\nFound ndims = %d instead.\n", ndims);
            goto except;
        }
        const int array_type = PyArray_TYPE(array);
        switch (array_type) {
            case NPY_FLOAT:
                if (element_size != sizeof(float)) {
                    snprintf(msg, 1024, "ERROR: Input coordinates are float32 but provided pair weight is float64. Please use the same size.\n");
                    goto except;
                }
                break;
            case NPY_DOUBLE:
                if (element_size != sizeof(double)) {
                    snprintf(msg, 1024, "ERROR: Input coordinates are float64 but provided pair weight is float32. Please use the same size.\n");
                    goto except;
                }
                break;
            default:
                snprintf(msg, 1024, "TypeError: Expected floating array for pair weight. Instead found type-num %d.\n", array_type);
                goto except;
        }
    }
    const int num = (int) PyArray_SIZE((PyArrayObject *) sep);
    const int num_weight = (int) PyArray_SIZE((PyArrayObject *) weight);
    if (num_weight != num) {
        snprintf(msg, 1024, "ERROR: Expected pair weight array and separation array to be of same size.\nFound %d and %d instead.\n", num_weight, num);
        goto except;
    }
    set_pair_weight_struct(pair_weight_st, (void *) PyArray_DATA(sep), (void *) PyArray_DATA(weight), num, pair_weight_st->noffset, pair_weight_st->default_value, pair_weight_st->correction_bits, pair_weight_st->num_bits);
    goto finally;
except:
    countpairs_error_out(module, msg);
    Py_XDECREF(sep);
    Py_XDECREF(weight);
    return EXIT_FAILURE;
finally:
    Py_XDECREF(sep);
    Py_XDECREF(weight);
    return status;
}


static int check_selection(PyObject *module, selection_struct *selection_st, PyObject *attrs_selection_obj)
{
    int status = EXIT_SUCCESS;
    char msg[1024];
    selection_st->selection_type = NONE_SELECTION;
    if (attrs_selection_obj != NULL) {
        if (!PyDict_Check(attrs_selection_obj)) {
            snprintf(msg, 1024, "Please input dict of name: limits (tuple) for attrs_selection");
            goto except;
        }
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(attrs_selection_obj, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "rp") == 0) {
                set_selection_struct(selection_st, RP_SELECTION, PyFloat_AsDouble(PySequence_Fast_GET_ITEM(value, 0)), PyFloat_AsDouble(PySequence_Fast_GET_ITEM(value, 1)));
            }
            else if (PyUnicode_CompareWithASCIIString(key, "theta") == 0) {
                set_selection_struct(selection_st, THETA_SELECTION, PyFloat_AsDouble(PySequence_Fast_GET_ITEM(value, 0)), PyFloat_AsDouble(PySequence_Fast_GET_ITEM(value, 1)));
            }
            else {
                snprintf(msg, 1024, "ERROR: Found unknown key in attrs_selection\n");
                goto except;
            }
        }
    }
    goto finally;
except:
    countpairs_error_out(module, msg);
    return EXIT_FAILURE;
finally:
    return status;
}


static int check_binarray(PyObject *module, binarray* bins, PyArrayObject *bins_obj) {

    char msg[1024];

    /* All the arrays should be 1-D*/
    const int ndims = PyArray_NDIM(bins_obj);

    if(ndims != 1) {
        snprintf(msg, 1024, "ERROR: Expected 1-D numpy arrays.\nFound ndims = %d instead", ndims);
        countpairs_error_out(module, msg);
        return EXIT_FAILURE;
    }
    return set_binarray(bins, (double *) PyArray_DATA(bins_obj), (int) PyArray_SIZE(bins_obj));
}


static PyObject *countpairs_countpairs(PyObject *self, PyObject *args, PyObject *kwargs)
{
    //Error-handling is global in python2 -> stored in struct module_state _struct declared at the top of this file
#if PY_MAJOR_VERSION < 3
    (void) self;
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash.
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif
    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    PyArrayObject *x2_obj=NULL, *y2_obj=NULL, *z2_obj=NULL;
    PyArrayObject *bins_obj=NULL;
    PyObject *weights1_obj=NULL, *weights2_obj=NULL;

    int autocorr=0;
    int nthreads=4;
    char *weighting_method_str = NULL;

    PyObject *pair_weight_obj=NULL, *sep_pair_weight_obj=NULL, *attrs_pair_weight_obj=NULL;

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.instruction_set = -1;
    options.periodic = 1;
    options.need_avg_sep = 0;
    options.c_api_timer = 0;
    options.copy_particles = 1;
    options.enable_min_sep_opt = 1;

    int8_t xbin_ref=options.bin_refine_factors[0],
        ybin_ref=options.bin_refine_factors[1],
        zbin_ref=options.bin_refine_factors[2];

    static char *kwlist[] = {
        "autocorr",
        "nthreads",
        "binfile",
        "X1",
        "Y1",
        "Z1",
        "weights1",
        "X2",
        "Y2",
        "Z2",
        "weights2",
        "periodic",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "boxsize",  // 3-tuple
        "output_ravg",
        "xbin_refine_factor",
        "ybin_refine_factor",
        "zbin_refine_factor",
        "max_cells_per_dim",
        "copy_particles",
        "enable_min_sep_opt",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX512F, AVX, SSE, FALLBACK */
        "weight_type",
        "pair_weights",
        "sep_pair_weights",
        "attrs_pair_weights",
        "bin_type",
        NULL
    };

    // Note: type 'O!' doesn't allow for None to be passed, which we might want to do.
    if ( ! PyArg_ParseTupleAndKeywords(args, kwargs, "iiO!O!O!O!|OO!O!O!Obb(ddd)bbbbhbbbisO!O!OI", kwlist,
                                       &autocorr,&nthreads,
                                       &PyArray_Type,&bins_obj,
                                       &PyArray_Type,&x1_obj,
                                       &PyArray_Type,&y1_obj,
                                       &PyArray_Type,&z1_obj,
                                       &weights1_obj,
                                       &PyArray_Type,&x2_obj,
                                       &PyArray_Type,&y2_obj,
                                       &PyArray_Type,&z2_obj,
                                       &weights2_obj,
                                       &(options.periodic),
                                       &(options.verbose),
                                       &(options.boxsize_x),
                                       &(options.boxsize_y),
                                       &(options.boxsize_z),
                                       &(options.need_avg_sep),
                                       &xbin_ref, &ybin_ref, &zbin_ref,
                                       &(options.max_cells_per_dim),
                                       &(options.copy_particles),
                                       &(options.enable_min_sep_opt),
                                       &(options.c_api_timer),
                                       &(options.instruction_set),
                                       &weighting_method_str,
                                       &PyArray_Type,&pair_weight_obj,
                                       &PyArray_Type,&sep_pair_weight_obj,
                                       &attrs_pair_weight_obj,
                                       &(options.bin_type))

         ) {

        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In DD> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }

        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }

    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }

    if(xbin_ref != options.bin_refine_factors[0] ||
       ybin_ref != options.bin_refine_factors[1] ||
       zbin_ref != options.bin_refine_factors[2]) {
        options.bin_refine_factors[0] = xbin_ref;
        options.bin_refine_factors[1] = ybin_ref;
        options.bin_refine_factors[2] = zbin_ref;
        set_bin_refine_scheme(&options, BINNING_CUST);//custom binning -> code will honor requested binning scheme
    }

    /* Validate the user's choice of weighting method */
    weight_method_t weighting_method;
    int wstatus = get_weight_method_by_name(weighting_method_str, &weighting_method);
    if(wstatus != EXIT_SUCCESS){
        char msg[1024];
        snprintf(msg, 1024, "ValueError: In %s: unknown weight_type \"%s\"!", __FUNCTION__, weighting_method_str);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }
    if(weighting_method == NONE){
        // Do not attempt to validate the weights array if it will not be used!
        weights1_obj = NULL;
        weights2_obj = NULL;
    }


    /* We have numpy arrays and all the required inputs*/
    /* How many data points are there? And are they all of floating point type */
    size_t element_size;
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return
        Py_RETURN_NONE;
    }
    struct extra_options extra = get_extra_options(weighting_method);

    /* Validate the user's choice of weighting method */
    /*int found_weights = weights1_obj == NULL ? 0 : PyArray_SHAPE(weights1_obj)[0];
    struct extra_options extra = get_extra_options(weighting_method);
    if(extra.weights0.num_weights > 0 && extra.weights0.num_weights != found_weights){
        char msg[1024];
        snprintf(msg, 1024, "ValueError: In %s: specified weighting method %s which requires %"PRId64" weight(s)-per-particle, but found %d weight(s) instead!\n",
                 __FUNCTION__, weighting_method_str, extra.weights0.num_weights, found_weights);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }

    if(extra.weights0.num_weights > 0 && found_weights > MAX_NUM_WEIGHTS){
        char msg[1024];
        snprintf(msg, 1024, "ValueError: In %s: Provided %d weights-per-particle, but the code was compiled with MAX_NUM_WEIGHTS=%d.\n",
                 __FUNCTION__, found_weights, MAX_NUM_WEIGHTS);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }*/

    int64_t ND2 = 0;
    if(autocorr == 0) {
        char msg[1024];
        if(x2_obj == NULL || y2_obj == NULL || z2_obj == NULL) {
            snprintf(msg, 1024, "ValueError: In %s: If autocorr is 0, need to pass the second set of positions (X2=numpy array, Y2=numpy array, Z2=numpy array).\n",
                     __FUNCTION__);
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
        if((weights1_obj == NULL) != (weights2_obj == NULL)){
            snprintf(msg, 1024, "ValueError: In %s: If autocorr is 0, must pass either zero or two sets of weights.\n",
                     __FUNCTION__);
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
        size_t element_size2;
        ND2 = check_dims_and_datatype(module, x2_obj, y2_obj, z2_obj, &element_size2);
        if(ND2 == -1) {
            //Error has already been set -> simply return
            Py_RETURN_NONE;
        }

        if(element_size != element_size2) {
            snprintf(msg, 1024, "TypeError: In %s: The two arrays must have the same data-type. First array is of type %s while second array is of type %s\n",
                     __FUNCTION__, element_size == 4 ? "floats":"doubles", element_size2 == 4 ? "floats":"doubles");
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
    }

    /*
       Interpret the input objects as numpy arrays (of whatever the input type the python object has).
       NULL initialization is necessary since we might be calling XDECREF.
       The input objects can be converted into the required DOUBLE array.
    */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    PyObject *y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    PyObject *z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);
    /*PyObject *weights1_array = NULL;
    if(weights1_obj != NULL){
        weights1_array = PyArray_FromArray(weights1_obj, NOTYPE_DESCR, requirements);
    }*/

    /* NULL initialization is necessary since we might be calling XDECREF*/
    PyObject *x2_array = NULL, *y2_array = NULL, *z2_array = NULL;
    if(autocorr == 0) {
        x2_array = PyArray_FromArray(x2_obj, NOTYPE_DESCR, requirements);
        y2_array = PyArray_FromArray(y2_obj, NOTYPE_DESCR, requirements);
        z2_array = PyArray_FromArray(z2_obj, NOTYPE_DESCR, requirements);
        /*if(weights2_obj != NULL){
            weights2_array = PyArray_FromArray(weights2_obj, NOTYPE_DESCR, requirements);
        }*/
    }

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL ||
        (autocorr==0 && (x2_array == NULL || y2_array == NULL || z2_array == NULL))) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);
        //Py_XDECREF(weights1_array);

        Py_XDECREF(x2_array);
        Py_XDECREF(y2_array);
        Py_XDECREF(z2_array);
        //Py_XDECREF(weights2_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert input to arrays of allowed floating point types (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }

    /* Get pointers to the data */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);
    /*void *weights1=NULL;
    if(weights1_array != NULL){
        weights1 = PyArray_DATA((PyArrayObject *) weights1_array);
    }*/
    if (weights1_obj != NULL) wstatus = check_weights(module, weights1_obj, &(extra.weights0), extra.weight_method, ND1, element_size);

    void *X2 = NULL, *Y2=NULL, *Z2=NULL;
    if(autocorr==0) {
        X2 = PyArray_DATA((PyArrayObject *) x2_array);
        Y2 = PyArray_DATA((PyArrayObject *) y2_array);
        Z2 = PyArray_DATA((PyArrayObject *) z2_array);
        /*if(weights2_array != NULL){
            weights2 = PyArray_DATA((PyArrayObject *) weights2_array);
        }*/
        if (weights2_obj != NULL) wstatus = check_weights(module, weights2_obj, &(extra.weights1), extra.weight_method, ND2, element_size);
    }
    wstatus = check_pair_weight(module, &(extra.pair_weight), sep_pair_weight_obj, pair_weight_obj, element_size, attrs_pair_weight_obj);

    /* Pack the weights into extra_options */
    /*for(int64_t w = 0; w < extra.weights0.num_weights; w++){
        extra.weights0.weights[w] = (char *) weights1 + w*ND1*element_size;
        if(autocorr == 0){
            extra.weights1.weights[w] = (char *) weights2 + w*ND2*element_size;
        }
    }*/
    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    binarray bins;
    wstatus = check_binarray(module, &bins, bins_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    results_countpairs results;
    options.float_type = element_size;
    double c_api_time = 0.0;
    int status = countpairs(ND1,X1,Y1,Z1,
                            ND2,X2,Y2,Z2,
                            nthreads,
                            autocorr,
                            &bins,
                            &results,
                            &options,
                            &extra);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);
    Py_XDECREF(x2_array);Py_XDECREF(y2_array);Py_XDECREF(z2_array);
    free_binarray(&bins);

    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    /* Build the output list */
    PyObject *ret = PyList_New(0);
    double rlow=results.rupp[0];
    for(int i=1;i<results.nbin;i++) {
        const double rpavg = results.rpavg[i];
        const double weight_avg = results.weightavg[i];
        PyObject *item = Py_BuildValue("(dddkd)", rlow,results.rupp[i],rpavg,results.npairs[i],weight_avg);
        PyList_Append(ret, item);
        Py_XDECREF(item);
        rlow=results.rupp[i];
    }

    free_results(&results);

    PyObject *rettuple = Py_BuildValue("(Od)", ret, c_api_time);
    Py_DECREF(ret);  // transfer reference ownership to the tuple
    return rettuple;
}


static PyObject *countpairs_countpairs_rp_pi(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash.
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif
    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    PyArrayObject *x2_obj=NULL, *y2_obj=NULL, *z2_obj=NULL;
    PyArrayObject *bins_obj=NULL;
    PyObject *weights1_obj=NULL, *weights2_obj=NULL;
    int autocorr=0;
    int nthreads=4;

    double pimax;
    int npibins;
    char *weighting_method_str = NULL;
    PyObject *pair_weight_obj=NULL, *sep_pair_weight_obj=NULL, *attrs_pair_weight_obj=NULL;

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.instruction_set = -1;
    options.periodic = 1;
    options.c_api_timer = 0;
    options.copy_particles = 1;
    options.enable_min_sep_opt = 1;
    int8_t xbin_ref=options.bin_refine_factors[0],
        ybin_ref=options.bin_refine_factors[1],
        zbin_ref=options.bin_refine_factors[2];

    static char *kwlist[] = {
        "autocorr",
        "nthreads",
        "binfile",
        "pimax",
        "npibins",
        "X1",
        "Y1",
        "Z1",
        "weights1",
        "X2",
        "Y2",
        "Z2",
        "weights2",
        "periodic",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "boxsize",  // 3-tuple
        "output_rpavg",
        "xbin_refine_factor",
        "ybin_refine_factor",
        "zbin_refine_factor",
        "max_cells_per_dim",
        "copy_particles",
        "enable_min_sep_opt",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX512F, AVX, SSE, FALLBACK */
        "weight_type",
        "pair_weights",
        "sep_pair_weights",
        "attrs_pair_weights",
        "bin_type",
        NULL
    };

    if ( ! PyArg_ParseTupleAndKeywords(args, kwargs, "iiO!diO!O!O!|OO!O!O!Obb(ddd)bbbbhbbbisO!O!OI", kwlist,
                                       &autocorr,&nthreads,
                                       &PyArray_Type,&bins_obj,
                                       &pimax,&npibins,
                                       &PyArray_Type,&x1_obj,
                                       &PyArray_Type,&y1_obj,
                                       &PyArray_Type,&z1_obj,
                                       &weights1_obj,
                                       &PyArray_Type,&x2_obj,
                                       &PyArray_Type,&y2_obj,
                                       &PyArray_Type,&z2_obj,
                                       &weights2_obj,
                                       &(options.periodic),
                                       &(options.verbose),
                                       &(options.boxsize_x),
                                       &(options.boxsize_y),
                                       &(options.boxsize_z),
                                       &(options.need_avg_sep),
                                       &xbin_ref, &ybin_ref, &zbin_ref,
                                       &(options.max_cells_per_dim),
                                       &(options.copy_particles),
                                       &(options.enable_min_sep_opt),
                                       &(options.c_api_timer),
                                       &(options.instruction_set),
                                       &weighting_method_str,
                                       &PyArray_Type,&pair_weight_obj,
                                       &PyArray_Type,&sep_pair_weight_obj,
                                       &attrs_pair_weight_obj,
                                       &(options.bin_type))

         ) {
        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In DDrppi> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }

        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }
    options.autocorr=autocorr;
    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }

    if(xbin_ref != options.bin_refine_factors[0] ||
       ybin_ref != options.bin_refine_factors[1] ||
       zbin_ref != options.bin_refine_factors[2]) {
        options.bin_refine_factors[0] = xbin_ref;
        options.bin_refine_factors[1] = ybin_ref;
        options.bin_refine_factors[2] = zbin_ref;
        set_bin_refine_scheme(&options, BINNING_CUST);//custom binning -> code will honor requested binning scheme
    }

    /* Validate the user's choice of weighting method */
    weight_method_t weighting_method;
    int wstatus = get_weight_method_by_name(weighting_method_str, &weighting_method);
    if(wstatus != EXIT_SUCCESS){
        char msg[1024];
        snprintf(msg, 1024, "ValueError: In %s: unknown weight_type \"%s\"!", __FUNCTION__, weighting_method_str);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }
    if(weighting_method == NONE){
        // Do not attempt to validate the weights array if it will not be used!f
        weights1_obj = NULL;
        weights2_obj = NULL;
    }

    size_t element_size;
    /* How many data points are there? And are they all of floating point type */
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return
        Py_RETURN_NONE;
    }
    struct extra_options extra = get_extra_options(weighting_method);

    int64_t ND2=ND1;
    if(autocorr == 0) {
        char msg[1024];
        if(x2_obj == NULL || y2_obj == NULL || z2_obj == NULL) {
            snprintf(msg, 1024, "ValueError: In %s: If autocorr is 0, need to pass the second set of positions (X2=numpy array, Y2=numpy array, Z2=numpy array).\n",
                     __FUNCTION__);
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
        if((weights1_obj == NULL) != (weights2_obj == NULL)){
            snprintf(msg, 1024, "ValueError: In %s: If autocorr is 0, must pass either zero or two sets of weights.\n",
                     __FUNCTION__);
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }

        size_t element_size2;
        ND2 = check_dims_and_datatype(module, x2_obj, y2_obj, z2_obj, &element_size2);
        if(ND2 == -1) {
            //Error has already been set -> simply return
            Py_RETURN_NONE;
        }

        if(element_size != element_size2) {
            snprintf(msg, 1024, "TypeError: In %s: The two arrays must have the same data-type. First array is of type %s while second array is of type %s\n",
                     __FUNCTION__, element_size == 4 ? "floats":"doubles", element_size2 == 4 ? "floats":"doubles");
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
    }

    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    PyObject *y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    PyObject *z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);

    PyObject *x2_array = NULL, *y2_array = NULL, *z2_array = NULL;
    if(autocorr == 0) {
        x2_array = PyArray_FromArray(x2_obj, NOTYPE_DESCR, requirements);
        y2_array = PyArray_FromArray(y2_obj, NOTYPE_DESCR, requirements);
        z2_array = PyArray_FromArray(z2_obj, NOTYPE_DESCR, requirements);
    }

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL ||
        (autocorr == 0 && (x2_array == NULL || y2_array == NULL || z2_array == NULL))) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);

        Py_XDECREF(x2_array);
        Py_XDECREF(y2_array);
        Py_XDECREF(z2_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert input to arrays of allowed floating point types (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }


    /* Get pointers to the data as C-types. */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    if (weights1_obj != NULL) wstatus = check_weights(module, weights1_obj, &(extra.weights0), extra.weight_method, ND1, element_size);

    void *X2 = NULL, *Y2 = NULL, *Z2 = NULL;
    if(autocorr == 0) {
        X2 = PyArray_DATA((PyArrayObject *) x2_array);
        Y2 = PyArray_DATA((PyArrayObject *) y2_array);
        Z2 = PyArray_DATA((PyArrayObject *) z2_array);

        if (weights2_obj != NULL) wstatus = check_weights(module, weights2_obj, &(extra.weights1), extra.weight_method, ND2, element_size);
    }
    wstatus = check_pair_weight(module, &(extra.pair_weight), sep_pair_weight_obj, pair_weight_obj, element_size, attrs_pair_weight_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }
    binarray bins;
    wstatus = check_binarray(module, &bins, bins_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    options.float_type = element_size;
    results_countpairs_rp_pi results;
    double c_api_time = 0.0;
    int status = countpairs_rp_pi(ND1,X1,Y1,Z1,
                                  ND2,X2,Y2,Z2,
                                  nthreads,
                                  autocorr,
                                  &bins,
                                  pimax,
                                  npibins,
                                  &results,
                                  &options,
                                  &extra);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);//x1 should absolutely not be NULL
    Py_XDECREF(x2_array);Py_XDECREF(y2_array);Py_XDECREF(z2_array);//x2 might be NULL depending on value of autocorr
    free_binarray(&bins);
    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }


    /* Build the output list */
    PyObject *ret = PyList_New(0);//create an empty list
    double rlow=results.rupp[0];
    const double dpi = 2.*pimax/(double)results.npibin ;

    for(int i=1;i<results.nbin;i++) {
        for(int j=0;j<results.npibin;j++) {
            const int bin_index = i*(results.npibin + 1) + j;
            const double rpavg = results.rpavg[bin_index];
            const double weight_avg = results.weightavg[bin_index];
            PyObject *item = Py_BuildValue("(ddddkd)", rlow,results.rupp[i],rpavg,(j+1)*dpi-pimax,results.npairs[bin_index], weight_avg);
            PyList_Append(ret, item);
            Py_XDECREF(item);
        }
        rlow=results.rupp[i];
    }
    free_results_rp_pi(&results);

    PyObject *rettuple = Py_BuildValue("(Od)", ret, c_api_time);
    Py_DECREF(ret);  // transfer reference ownership to the tuple
    return rettuple;
}


static PyObject *countpairs_countpairs_wp(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;//to suppress the unused variable warning. Terrible hack
    PyObject *module = NULL;//need not be used -> setting to NULL so any attempts to dereference will result in a crash.
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif
    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    PyArrayObject *bins_obj=NULL;
    PyObject *weights1_obj=NULL;
    double boxsize,pimax;
    int nthreads=1;
    char *weighting_method_str = NULL;
    PyObject *pair_weight_obj=NULL, *sep_pair_weight_obj=NULL, *attrs_pair_weight_obj=NULL;
    size_t element_size;

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.instruction_set = -1;
    options.need_avg_sep = 0;
    options.periodic = 1;
    options.copy_particles = 1;
    options.enable_min_sep_opt = 1;
    options.c_api_timer = 0;
    options.c_cell_timer = 0;
    int8_t xbin_ref=options.bin_refine_factors[0],
        ybin_ref=options.bin_refine_factors[1],
        zbin_ref=options.bin_refine_factors[2];

    static char *kwlist[] = {
        "boxsize",
        "nthreads",
        "binfile",
        "pimax",
        "X",
        "Y",
        "Z",
        "weights",
        "weight_type",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "output_rpavg",
        "xbin_refine_factor",
        "ybin_refine_factor",
        "zbin_refine_factor",
        "max_cells_per_dim",
        "copy_particles",
        "enable_min_sep_opt",
        "c_api_timer",
        "c_cell_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX512F, AVX, SSE, FALLBACK */
        "pair_weights",
        "sep_pair_weights",
        "attrs_pair_weights",
        "bin_type",
        NULL
    };

    if( ! PyArg_ParseTupleAndKeywords(args, kwargs, "diO!dO!O!O!|OsbbbbbhbbbbiO!O!OI", kwlist,
                                      &boxsize,&nthreads,
                                      &PyArray_Type,&bins_obj,
                                      &pimax,
                                      &PyArray_Type,&x1_obj,
                                      &PyArray_Type,&y1_obj,
                                      &PyArray_Type,&z1_obj,
                                      &weights1_obj,
                                      &weighting_method_str,
                                      &(options.verbose),
                                      &(options.need_avg_sep),
                                      &xbin_ref, &ybin_ref, &zbin_ref,
                                      &(options.max_cells_per_dim),
                                      &(options.copy_particles),
                                      &(options.enable_min_sep_opt),
                                      &(options.c_api_timer),
                                      &(options.c_cell_timer),
                                      &(options.instruction_set),
                                      &PyArray_Type,&pair_weight_obj,
                                      &PyArray_Type,&sep_pair_weight_obj,
                                      &attrs_pair_weight_obj,
                                      &(options.bin_type))

        ){
        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In wp> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }

        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }
    options.boxsize=boxsize;

    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }

    if(xbin_ref != options.bin_refine_factors[0] ||
       ybin_ref != options.bin_refine_factors[1] ||
       zbin_ref != options.bin_refine_factors[2]) {
        options.bin_refine_factors[0] = xbin_ref;
        options.bin_refine_factors[1] = ybin_ref;
        options.bin_refine_factors[2] = zbin_ref;
        set_bin_refine_scheme(&options, BINNING_CUST);//custom binning -> code will honor requested binning scheme
    }

    /* Validate the user's choice of weighting method */
    weight_method_t weighting_method;
    int wstatus = get_weight_method_by_name(weighting_method_str, &weighting_method);
    if(wstatus != EXIT_SUCCESS){
        char msg[1024];
        snprintf(msg, 1024, "ValueError: In %s: unknown weight_type \"%s\"!", __FUNCTION__, weighting_method_str);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }
    if(weighting_method == NONE){
        // Do not attempt to validate the weights array if it will not be used!
        weights1_obj = NULL;
    }

    /* How many data points are there? And are they all of floating point type */
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return
        Py_RETURN_NONE;
    }
    struct extra_options extra = get_extra_options(weighting_method);

    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    PyObject *y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    PyObject *z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert input array to allowed floating point types (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        perror(NULL);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }


    /* Get pointers to the data as C-types. */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    if (weights1_obj != NULL) wstatus = check_weights(module, weights1_obj, &(extra.weights0), extra.weight_method, ND1, element_size);
    wstatus = check_pair_weight(module, &(extra.pair_weight), sep_pair_weight_obj, pair_weight_obj, element_size, attrs_pair_weight_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }
    binarray bins;
    wstatus = check_binarray(module, &bins, bins_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    results_countpairs_wp results;
    options.float_type = element_size;
    double c_api_time = 0.0;
    int status = countpairs_wp(ND1,X1,Y1,Z1,
                               boxsize,
                               nthreads,
                               &bins,
                               pimax,
                               &results,
                               &options,
                               &extra);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);
    free_binarray(&bins);

    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

#if 0
    for(int i=1;i<results.nbin;i++) {
        const double rpavg = results.rpavg[i];
        fprintf(stderr,"%lf %lf %lf %lf %"PRIu64" %lf\n",results.rupp[i-1],results.rupp[i],rpavg,results.wp[i],results.npairs[i], results.weightavg[i]);
    }
#endif

    /* Build the output list */
    PyObject *ret = PyList_New(0);
    double rlow=results.rupp[0];
    for(int i=1;i<results.nbin;i++) {
        const double rpavg = results.rpavg[i];
        const double weight_avg = results.weightavg[i];
        PyObject *item = Py_BuildValue("(ddddkd)", rlow,results.rupp[i],rpavg,results.wp[i],results.npairs[i], weight_avg);
        PyList_Append(ret, item);
        Py_XDECREF(item);
        rlow=results.rupp[i];
    }
    free_results_wp(&results);

    PyObject *c_cell_time=PyList_New(0);
    if(options.c_cell_timer) {
        struct api_cell_timings *t = options.cell_timings;
        for(int i=0;i<options.totncells_timings;i++) {
            PyObject *item = Py_BuildValue("(kkkiii)", t->N1, t->N2, t->time_in_ns, t->first_cellindex, t->second_cellindex, t->tid);
            PyList_Append(c_cell_time, item);
            Py_XDECREF(item);
            t++;
        }
        free_cell_timings(&options);
    }
    PyObject *rettuple = Py_BuildValue("(OdO)", ret, c_api_time, c_cell_time);
    Py_DECREF(ret);  // transfer reference ownership to the tuple
    Py_DECREF(c_cell_time);
    return rettuple;
}


static PyObject *countpairs_countpairs_xi(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;//to suppress the unused variable warning. Terrible hack
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash.
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif

    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    PyArrayObject *bins_obj=NULL;
    PyObject *weights1_obj=NULL;
    double boxsize;
    int nthreads=4;
    char *weighting_method_str = NULL;
    PyObject *pair_weight_obj=NULL, *sep_pair_weight_obj=NULL, *attrs_pair_weight_obj=NULL;

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.periodic=1;
    options.instruction_set = -1; //from enum
    options.c_api_timer = 0;
    options.copy_particles = 1;
    options.enable_min_sep_opt = 1;
    int8_t xbin_ref=options.bin_refine_factors[0],
        ybin_ref=options.bin_refine_factors[1],
        zbin_ref=options.bin_refine_factors[2];

    static char *kwlist[] = {
        "boxsize",
        "nthreads",
        "binfile",
        "X",
        "Y",
        "Z",
        "weights",
        "weight_type",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "output_ravg",
        "xbin_refine_factor",
        "ybin_refine_factor",
        "zbin_refine_factor",
        "max_cells_per_dim",
        "copy_particles",
        "enable_min_sep_opt",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX512F, AVX, SSE, FALLBACK */
        "pair_weights",
        "sep_pair_weights",
        "attrs_pair_weights",
        "bin_type",
        NULL
    };


    if( ! PyArg_ParseTupleAndKeywords(args, kwargs, "diO!O!O!O!|OsbbbbbhbbbiO!O!OI", kwlist,
                                      &boxsize,&nthreads,
                                      &PyArray_Type,&bins_obj,
                                      &PyArray_Type,&x1_obj,
                                      &PyArray_Type,&y1_obj,
                                      &PyArray_Type,&z1_obj,
                                      &weights1_obj,
                                      &weighting_method_str,
                                      &(options.verbose),
                                      &(options.need_avg_sep),
                                      &xbin_ref, &ybin_ref, &zbin_ref,
                                      &(options.max_cells_per_dim),
                                      &(options.copy_particles),
                                      &(options.enable_min_sep_opt),
                                      &(options.c_api_timer),
                                      &(options.instruction_set),
                                      &PyArray_Type,&pair_weight_obj,
                                      &PyArray_Type,&sep_pair_weight_obj,
                                      &attrs_pair_weight_obj,
                                      &(options.bin_type))
        ) {

        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In xi> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }

        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }

    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }
    if(xbin_ref != options.bin_refine_factors[0] ||
       ybin_ref != options.bin_refine_factors[1] ||
       zbin_ref != options.bin_refine_factors[2]) {
        options.bin_refine_factors[0] = xbin_ref;
        options.bin_refine_factors[1] = ybin_ref;
        options.bin_refine_factors[2] = zbin_ref;
        set_bin_refine_scheme(&options, BINNING_CUST);//custom binning -> code will honor requested binning scheme
    }


    /* Validate the user's choice of weighting method */
    weight_method_t weighting_method;
    int wstatus = get_weight_method_by_name(weighting_method_str, &weighting_method);
    if(wstatus != EXIT_SUCCESS){
        char msg[1024];
        snprintf(msg, 1024, "ValueError: In %s: unknown weight_type \"%s\"!", __FUNCTION__, weighting_method_str);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }
    if(weighting_method == NONE){
        // Do not attempt to validate the weights array if it will not be used!
        weights1_obj = NULL;
    }

    /* How many data points are there? And are they all of floating point type */
    size_t element_size;
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return
        Py_RETURN_NONE;
    }
    struct extra_options extra = get_extra_options(weighting_method);

    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    PyObject *y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    PyObject *z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert to array of allowed floating point type (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }

    /* Get pointers to the data as C-types. */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    if (weights1_obj != NULL) wstatus = check_weights(module, weights1_obj, &(extra.weights0), extra.weight_method, ND1, element_size);
    wstatus = check_pair_weight(module, &(extra.pair_weight), sep_pair_weight_obj, pair_weight_obj, element_size, attrs_pair_weight_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }
    binarray bins;
    wstatus = check_binarray(module, &bins, bins_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    results_countpairs_xi results;
    options.periodic = 1;
    options.float_type = element_size;
    double c_api_time=0.0;
    int status = countpairs_xi(ND1,X1,Y1,Z1,
                               boxsize,
                               nthreads,
                               &bins,
                               &results,
                               &options,
                               &extra);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);
    free_binarray(&bins);
    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }


#if 0
    for(int i=1;i<results.nbin;i++) {
        const double rpavg = results.rpavg[i];
        fprintf(stderr,"%lf %lf %lf %lf %"PRIu64"\n",results.rupp[i-1],results.rupp[i],rpavg,results.xi[i],results.npairs[i]);
    }
#endif

    /* Build the output list */
    PyObject *ret = PyList_New(0);
    double rlow=results.rupp[0];
    for(int i=1;i<results.nbin;i++) {
        const double ravg = results.ravg[i];
        const double weight_avg = results.weightavg[i];
        PyObject *item = Py_BuildValue("(ddddkd)", rlow,results.rupp[i],ravg,results.xi[i],results.npairs[i], weight_avg);
        PyList_Append(ret, item);
        Py_XDECREF(item);
        rlow=results.rupp[i];
    }
    free_results_xi(&results);

    PyObject *rettuple = Py_BuildValue("(Od)", ret, c_api_time);
    Py_DECREF(ret);  // transfer reference ownership to the tuple
    return rettuple;
}


static PyObject *countpairs_countpairs_s_mu(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash.
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif
    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    PyArrayObject *x2_obj=NULL, *y2_obj=NULL, *z2_obj=NULL;
    PyArrayObject *bins_obj=NULL;
    PyObject *weights1_obj=NULL, *weights2_obj=NULL;
    int autocorr=0;
    int nthreads=4;

    double mu_max;
    int nmu_bins;
    char *weighting_method_str = NULL;
    PyObject *pair_weight_obj=NULL, *sep_pair_weight_obj=NULL, *attrs_pair_weight_obj=NULL, *attrs_selection_obj=NULL;

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.instruction_set = -1;
    options.periodic = 1;
    options.c_api_timer = 0;
    options.copy_particles = 1;
    options.enable_min_sep_opt = 1;
    options.fast_divide_and_NR_steps = 0;
    int8_t xbin_ref=options.bin_refine_factors[0],
        ybin_ref=options.bin_refine_factors[1],
        zbin_ref=options.bin_refine_factors[2];

    static char *kwlist[] = {
        "autocorr",
        "nthreads",
        "binfile",
        "mu_max",
        "nmu_bins",
        "X1",
        "Y1",
        "Z1",
        "weights1",
        "X2",
        "Y2",
        "Z2",
        "weights2",
        "periodic",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "boxsize",  // 3-tuple
        "output_savg",
        "fast_divide_and_NR_steps",
        "xbin_refine_factor",
        "ybin_refine_factor",
        "zbin_refine_factor",
        "max_cells_per_dim",
        "copy_particles",
        "enable_min_sep_opt",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX512F, AVX, SSE, FALLBACK */
        "gpu",
        "weight_type",
        "pair_weights",
        "sep_pair_weights",
        "attrs_pair_weights",
        "attrs_selection",
        "bin_type",
        NULL
    };

    if ( ! PyArg_ParseTupleAndKeywords(args, kwargs, "iiO!diO!O!O!|OO!O!O!Obb(ddd)bbbbbhbbbiisO!O!OOI", kwlist,
                                       &autocorr,&nthreads,
                                       &PyArray_Type,&bins_obj,
                                       &mu_max, &nmu_bins,
                                       &PyArray_Type,&x1_obj,
                                       &PyArray_Type,&y1_obj,
                                       &PyArray_Type,&z1_obj,
                                       &weights1_obj,
                                       &PyArray_Type,&x2_obj,
                                       &PyArray_Type,&y2_obj,
                                       &PyArray_Type,&z2_obj,
                                       &weights2_obj,
                                       &(options.periodic),
                                       &(options.verbose),
                                       &(options.boxsize_x),
                                       &(options.boxsize_y),
                                       &(options.boxsize_z),
                                       &(options.need_avg_sep),
                                       &(options.fast_divide_and_NR_steps),
                                       &xbin_ref, &ybin_ref, &zbin_ref,
                                       &(options.max_cells_per_dim),
                                       &(options.copy_particles),
                                       &(options.enable_min_sep_opt),
                                       &(options.c_api_timer),
                                       &(options.instruction_set),
                                       &(options.use_gpu),
                                       &weighting_method_str,
                                       &PyArray_Type,&pair_weight_obj,
                                       &PyArray_Type,&sep_pair_weight_obj,
                                       &attrs_pair_weight_obj,
                                       &attrs_selection_obj,
                                       &(options.bin_type))

         ) {
        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In %s> Could not parse the arguments. Input parameters are: \n", __FUNCTION__);

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }

        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }
    options.autocorr=autocorr;
    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }

    if(xbin_ref != options.bin_refine_factors[0] ||
       ybin_ref != options.bin_refine_factors[1] ||
       zbin_ref != options.bin_refine_factors[2]) {
        options.bin_refine_factors[0] = xbin_ref;
        options.bin_refine_factors[1] = ybin_ref;
        options.bin_refine_factors[2] = zbin_ref;
        set_bin_refine_scheme(&options, BINNING_CUST);//custom binning -> code will honor requested binning scheme
    }

    /* Validate the user's choice of weighting method */
    weight_method_t weighting_method;
    int wstatus = get_weight_method_by_name(weighting_method_str, &weighting_method);
    if(wstatus != EXIT_SUCCESS){
        char msg[1024];
        snprintf(msg, 1024, "ValueError: In %s: unknown weight_type \"%s\"!", __FUNCTION__, weighting_method_str);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }
    if(weighting_method == NONE){
        // Do not attempt to validate the weights array if it will not be used!
        weights1_obj = NULL;
        weights2_obj = NULL;
    }

    size_t element_size;
    /* How many data points are there? And are they all of floating point type */
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return
        Py_RETURN_NONE;
    }
    struct extra_options extra = get_extra_options(weighting_method);

    int64_t ND2=ND1;
    if(autocorr == 0) {
        char msg[1024];
        if(x2_obj == NULL || y2_obj == NULL || z2_obj == NULL) {
            snprintf(msg, 1024, "ValueError: In %s: If autocorr is 0, need to pass the second set of positions (X2=numpy array, Y2=numpy array, Z2=numpy array).\n",
                     __FUNCTION__);
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
        if((weights1_obj == NULL) != (weights2_obj == NULL)){
            snprintf(msg, 1024, "ValueError: In %s: If autocorr is 0, must pass either zero or two sets of weights.\n",
                     __FUNCTION__);
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }

        size_t element_size2;
        ND2 = check_dims_and_datatype(module, x2_obj, y2_obj, z2_obj, &element_size2);
        if(ND2 == -1) {
            //Error has already been set -> simply return
            Py_RETURN_NONE;
        }

        if(element_size != element_size2) {
            snprintf(msg, 1024, "TypeError: In %s: The two arrays must have the same data-type. First array is of type %s while second array is of type %s\n",
                     __FUNCTION__, element_size == 4 ? "floats":"doubles", element_size2 == 4 ? "floats":"doubles");
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
    }

    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    PyObject *y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    PyObject *z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);

    PyObject *x2_array = NULL, *y2_array = NULL, *z2_array = NULL;
    if(autocorr == 0) {
        x2_array = PyArray_FromArray(x2_obj, NOTYPE_DESCR, requirements);
        y2_array = PyArray_FromArray(y2_obj, NOTYPE_DESCR, requirements);
        z2_array = PyArray_FromArray(z2_obj, NOTYPE_DESCR, requirements);
    }

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL ||
        (autocorr == 0 && (x2_array == NULL || y2_array == NULL || z2_array == NULL))) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);

        Py_XDECREF(x2_array);
        Py_XDECREF(y2_array);
        Py_XDECREF(z2_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert input to arrays of allowed floating point types (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }


    /* Get pointers to the data as C-types. */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    if (weights1_obj != NULL) wstatus = check_weights(module, weights1_obj, &(extra.weights0), extra.weight_method, ND1, element_size);

    void *X2 = NULL, *Y2 = NULL, *Z2 = NULL;
    if(autocorr == 0) {
        X2 = PyArray_DATA((PyArrayObject *) x2_array);
        Y2 = PyArray_DATA((PyArrayObject *) y2_array);
        Z2 = PyArray_DATA((PyArrayObject *) z2_array);

        if (weights2_obj != NULL) wstatus = check_weights(module, weights2_obj, &(extra.weights1), extra.weight_method, ND2, element_size);
    }
    wstatus = check_pair_weight(module, &(extra.pair_weight), sep_pair_weight_obj, pair_weight_obj, element_size, attrs_pair_weight_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    wstatus = check_selection(module, &(options.selection), attrs_selection_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    binarray bins;
    wstatus = check_binarray(module, &bins, bins_obj);

    if(wstatus != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    options.float_type = element_size;
    results_countpairs_s_mu results;
    double c_api_time = 0.0;
    int status = countpairs_s_mu(ND1,X1,Y1,Z1,
                                 ND2,X2,Y2,Z2,
                                 nthreads,
                                 autocorr,
                                 &bins,
                                 mu_max,
                                 nmu_bins,
                                 &results,
                                 &options,
                                 &extra);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);//x1 should absolutely not be NULL
    Py_XDECREF(x2_array);Py_XDECREF(y2_array);Py_XDECREF(z2_array);//x2 might be NULL depending on value of autocorr
    free_binarray(&bins);
    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }


    /* Build the output list */
    PyObject *ret = PyList_New(0);//create an empty list
    double smin = results.supp[0];
    const double dmu = 2.*mu_max/(double)nmu_bins;//mu_min is assumed to be 0.0
    for(int i=1;i<results.nsbin;i++) {
        const double smax=results.supp[i];
        for(int j=0;j<results.nmu_bins;j++) {
            const int bin_index = i*(results.nmu_bins + 1) + j;
            const double savg = results.savg[bin_index];
            const double weight_avg = results.weightavg[bin_index];
            PyObject *item = Py_BuildValue("(ddddkd)", smin, smax, savg, (j+1)*dmu-mu_max, results.npairs[bin_index], weight_avg);
            PyList_Append(ret, item);
            Py_XDECREF(item);
        }
        smin = smax;
    }
    free_results_s_mu(&results);

    PyObject *rettuple = Py_BuildValue("(Od)", ret, c_api_time);
    Py_DECREF(ret);  // transfer reference ownership to the tuple
    return rettuple;
}


static PyObject *countpairs_countspheres_vpf(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;//to suppress the unused variable warning. Terrible hack
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash.
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif

    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    double rmax;
    int nbin,nc,num_pN;
    unsigned long seed=-1;

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.periodic = 1;
    options.instruction_set = -1;
    options.copy_particles = 1;
    options.c_api_timer = 0;

    /* Reset the bin refine factors default (since the VPF is symmetric in XYZ, conceptually the binning should be identical in all three directions)*/
    int bin_ref[] = {1,1,1};
    set_bin_refine_factors(&options, bin_ref);

    int8_t xbin_ref=options.bin_refine_factors[0],
        ybin_ref=options.bin_refine_factors[1],
        zbin_ref=options.bin_refine_factors[2];

    static char *kwlist[] = {
        "rmax",
        "nbins",
        "nspheres",
        "num_pN",
        "seed",
        "X",
        "Y",
        "Z",
        "periodic",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "boxsize",  // 3-tuple
        "xbin_refine_factor",
        "ybin_refine_factor",
        "zbin_refine_factor",
        "max_cells_per_dim",
        "copy_particles",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX512F, AVX, SSE, FALLBACK */
        NULL
    };

    if( ! PyArg_ParseTupleAndKeywords(args, kwargs,
                                      "diiikO!O!O!|bb(ddd)bbbhbbi", kwlist,
                                      &rmax,&nbin,&nc,&num_pN,&seed,
                                      &PyArray_Type,&x1_obj,
                                      &PyArray_Type,&y1_obj,
                                      &PyArray_Type,&z1_obj,
                                      &(options.periodic),
                                      &(options.verbose),
                                      &(options.boxsize_x),
                                      &(options.boxsize_y),
                                      &(options.boxsize_z),
                                      &xbin_ref, &ybin_ref, &zbin_ref,
                                      &(options.max_cells_per_dim),
                                      &(options.copy_particles),
                                      &(options.c_api_timer),
                                      &(options.instruction_set))

        ) {

        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In vpf> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }

        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }
    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }
    if(xbin_ref != options.bin_refine_factors[0] ||
       ybin_ref != options.bin_refine_factors[1] ||
       zbin_ref != options.bin_refine_factors[2]) {
        options.bin_refine_factors[0] = xbin_ref;
        options.bin_refine_factors[1] = ybin_ref;
        options.bin_refine_factors[2] = zbin_ref;
        set_bin_refine_scheme(&options, BINNING_CUST);//custom binning -> code will honor requested binning scheme
    }

    /* How many data points are there? And are they all of floating point type */
    size_t element_size;
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return
        Py_RETURN_NONE;
    }

    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    PyObject *y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    PyObject *z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert to array of allowed floating point type (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }

    /* Get pointers to the data as C-types. */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    /* Do the VPF calculation */
    results_countspheres results;
    options.float_type = element_size;
    double c_api_time=0.0;
    int status = countspheres(ND1, X1, Y1, Z1,
                              rmax, nbin, nc,
                              num_pN,
                              seed,
                              &results,
                              &options, NULL);

    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);
    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    /* Build the output list (of lists, since num_pN is determined at runtime) */
    PyObject *ret = PyList_New(0);
    const double rstep = rmax/(double)nbin ;
    for(int ibin=0;ibin<results.nbin;ibin++) {
        const double r=(ibin+1)*rstep;
        PyObject *item = PyList_New(0);
        PyObject *this_val = Py_BuildValue("d",r);
        PyList_Append(item, this_val);
        Py_XDECREF(this_val);
        for(int i=0;i<num_pN;i++) {
            this_val = Py_BuildValue("d",(results.pN)[ibin][i]);
            PyList_Append(item, this_val);
            Py_XDECREF(this_val);
        }
        PyList_Append(ret, item);
        Py_XDECREF(item);
    }

    free_results_countspheres(&results);

    PyObject *rettuple = Py_BuildValue("(Od)", ret, c_api_time);
    Py_DECREF(ret);  // transfer reference ownership to the tuple
    return rettuple;
}
