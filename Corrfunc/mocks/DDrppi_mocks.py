#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python wrapper around the C extension for the pair counter in
``mocks/DDrppi_mocks/``. This python wrapper is
:py:mod:`Corrfunc.mocks.DDrppi_mocks`
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__author__ = ('Manodeep Sinha')
__all__ = ('DDrppi_mocks', )


def DDrppi_mocks(autocorr, nthreads, binfile, pimax, npibins,
                 X1, Y1, Z1, weights1=None,
                 X2=None, Y2=None, Z2=None, weights2=None,
                 verbose=False, output_rpavg=False,
                 fast_divide_and_NR_steps=0,
                 xbin_refine_factor=2, ybin_refine_factor=2,
                 zbin_refine_factor=1, max_cells_per_dim=100,
                 copy_particles=True, enable_min_sep_opt=True,
                 c_api_timer=False, isa='fastest',
                 weight_type=None, bin_type='custom', los_type='midpoint',
                 pair_weights=None, sep_pair_weights=None, attrs_pair_weights=None):
    """
    Calculate the pair-counts corresponding to the 2-D correlation
    function, :math:`\\xi(r_p, \\pi)`. Pairs which are separated by less
    than the ``rp`` bins (specified in ``binfile``) in the
    X-Y plane, and less than ``pimax`` in the Z-dimension are
    counted. The input positions are expected to be on-sky co-ordinates.
    This module is suitable for calculating correlation functions for mock
    catalogs.

    If ``weights`` are provided, the resulting pair counts are weighted.  The
    weighting scheme depends on ``weight_type``.

    Returns a numpy structured array containing the pair counts for the
    specified bins.


    .. note:: that this module only returns pair counts and not the actual
       correlation function :math:`\\xi(r_p, \\pi)` or :math:`wp(r_p)`. See the
       utilities :py:mod:`Corrfunc.utils.convert_3d_counts_to_cf` and
       :py:mod:`Corrfunc.utils.convert_rp_pi_counts_to_wp` for computing
       :math:`\\xi(r_p, \\pi)` and :math:`wp(r_p)` respectively from the
       pair counts.


    Parameters
    ----------
    autocorr : boolean, required
        Boolean flag for auto/cross-correlation. If autocorr is set to 1,
        then the second set of particle positions are not required.

    nthreads : integer
        The number of OpenMP threads to use. Has no effect if OpenMP was not
        enabled during library compilation.

    binfile : string or an list/array of floats
        For string input: filename specifying the ``rp`` bins for
        ``DDrppi_mocks``. The file should contain white-space separated values
        of (rpmin, rpmax)  for each ``rp`` wanted. The bins need to be
        contiguous and sorted in increasing order (smallest bins come first).

        For array-like input: A sequence of ``rp`` values that provides the
        bin-edges. For example,
        ``np.logspace(np.log10(0.1), np.log10(10.0), 15)`` is a valid
        input specifying **14** (logarithmic) bins between 0.1 and 10.0. This
        array does not need to be sorted.

    pimax : double
        A double-precision value for the maximum separation along
        the Z-dimension.
        Note: pairs with :math:`-\\pi_{max} < \\pi < \\pi_{max}`
        (exclusive on both ends) are counted.

    npibins : int
        The number of linear ``pi`` bins, with the bins ranging from
        from (:math:`-\\pi_{max}`, :math:`\\pi_{max}`).

    X1/Y1/Z1 : array_like, real (float/double)
        The array of X/Y/Z positions for the first set of points.
        Calculations are done in the precision of the supplied arrays.

    weights1 : array_like, real (float/double), optional
        A scalar, or an array of weights of shape (n_weights, n_positions) or (n_positions,).
        `weight_type` specifies how these weights are used; results are returned
        in the `weightavg` field.  If only one of weights1 and weights2 is
        specified, the other will be set to uniform weights.

    X2 : array-like, real (float/double)
        The array of Right Ascensions for the second set of points. X's
        are expected to be in [0.0, 360.0], but the code will try to fix cases
        where the X's are in [-180, 180.0]. For peace of mind, always supply
        X's in [0.0, 360.0].

        Must be of same precision type as X1/Y1/Z1.

    Y2 : array-like, real (float/double)
        Array of Declinations for the second set of points. Y's are expected
        to be in the [-90.0, 90.0], but the code will try to fix cases where
        the Y's are in [0.0, 180.0]. Again, for peace of mind, always supply
        Y's in [-90.0, 90.0].

        Must be of same precision type as X1/Y1/Z1.

    Z2 : array-like, real (float/double)
        Array of (Speed Of Light * Redshift) values for the second set of
        points. Code will try to detect cases where ``redshifts`` have been
        passed and multiply the entire array with the ``speed of light``.

        Must be of same precision type as X1/Y1/Z1.

    weights2 : array-like, real (float/double), optional
        Same as weights1, but for the second set of positions

    verbose : boolean (default false)
        Boolean flag to control output of informational messages

    output_rpavg : boolean (default false)
        Boolean flag to output the average ``rp`` for each bin. Code will
        run slower if you set this flag.

        If you are calculating in single-precision, ``rpavg`` will suffer
        suffer from numerical loss of precision and can not be trusted. If
        you need accurate ``rpavg`` values, then pass in double precision
        arrays for the particle positions.

    fast_divide_and_NR_steps : integer (default 0)
        Replaces the division in ``AVX`` implementation with an approximate
        reciprocal, followed by ``fast_divide_and_NR_steps`` of Newton-Raphson.
        Can improve runtime by ~15-20% on older computers. Value of 0 uses
        the standard division operation.

    (xyz)bin_refine_factor : integer, default is (2,2,1); typically within [1-3]
        Controls the refinement on the cell sizes. Can have up to a 20% impact
        on runtime.

    max_cells_per_dim : integer, default is 100, typical values in [50-300]
        Controls the maximum number of cells per dimension. Total number of
        cells can be up to (max_cells_per_dim)^3. Only increase if ``rpmax`` is
        too small relative to the boxsize (and increasing helps the runtime).

    copy_particles : boolean (default True)
        Boolean flag to make a copy of the particle positions
        If set to False, the particles will be re-ordered in-place

        .. versionadded:: 2.3.0

    enable_min_sep_opt : boolean (default true)
       Boolean flag to allow optimizations based on min. separation between
       pairs of cells. Here to allow for comparison studies.

       .. versionadded:: 2.3.0

    c_api_timer : boolean (default false)
        Boolean flag to measure actual time spent in the C libraries. Here
        to allow for benchmarking and scaling studies.

    isa: string, case-insensitive (default ``fastest``)
       Controls the runtime dispatch for the instruction set to use. Possible
       options are: [``fastest``, ``avx512f``, ``avx``, ``sse42``, ``fallback``]

       Setting isa to ``fastest`` will pick the fastest available instruction
       set on the current computer. However, if you set ``isa`` to, say,
       ``avx`` and ``avx`` is not available on the computer, then the code will
       revert to using ``fallback`` (even though ``sse42`` might be available).

       Unless you are benchmarking the different instruction sets, you should
       always leave ``isa`` to the default value. And if you *are*
       benchmarking, then the string supplied here gets translated into an
       ``enum`` for the instruction set defined in ``utils/defs.h``.

    weight_type : string, optional (default None)
        The type of weighting to apply. One of ["pair_product", "inverse_bitwise", None].

    bin_type : string, case-insensitive (default ``custom``)
        Set to ``lin`` for speed-up in case of linearly-spaced bins.
        In this case, the bin number for a pair separated by ``r_p`` is given by
        ``(r_p - binfile[0])/(binfile[-1] - binfile[0])*(len(binfile) - 1)``,
        i.e. only the first and last bins of input ``binfile`` are considered.
        Then setting ``output_rpavg`` is virtually costless.
        For non-linear binning, set to 'custom'.
        In the vast majority of cases, bin_type='linear' will yield identical
        results to custom linear binning but with higher performance.
        In a few rare cases where a pair falls on a bin boundary,
        'linear' and custom linear may disagree on which bin the pair falls into
        due to finite floating point precision.
        ``auto`` will choose linear binning if input ``binfile`` is within
        ``rtol = 1e-05`` *and* ``atol = 1e-08`` (relative and absolute tolerance)
        of ``np.linspace(binfile[0], binfile[-1], len(binfile))``.

    los_type : string, case-insensitive (default ``midpoint``)
        Choice of line-of-sight :math:`d`:
        - "midpoint": :math:`d = \hat{r_{1} + r_{2}}`
        - "firstpoint": :math:`d = \hat{r_{1}}`

    pair_weights : array-like, optional. Default: None.
        Array of pair weights.

    sep_pair_weights : array-like, optional. Default: None.
        Array of separations corresponding to ``pair_weights``.

    attrs_pair_weights : dict. Default: None.
        Attributes for pair weights; in case ``weight_type`` is "inverse_bitwise",
        the dictionary of {"noffset": offset to be added to the bitwise counts,
        "default_value": default weight value if denominator is zero}.

    Returns
    -------
    results : Numpy structured array

        A numpy structured array containing [rpmin, rpmax, rpavg, pimax,
        npairs, weightavg] for each radial bin specified in the ``binfile``.
        If ``output_ravg`` is not set, then ``rpavg`` will be set to 0.0 for
        all bins; similarly for ``weightavg``. ``npairs`` contains the number
        of pairs in that bin and can be used to compute the actual
        :math:`\\xi(r_p, \\pi)` or :math:`wp(rp)` by combining with
        (DR, RR) counts.

    api_time : float, optional
        Only returned if ``c_api_timer`` is set.  ``api_time`` measures only
        the time spent within the C library and ignores all python overhead.

    Example
    -------
    >>> from __future__ import print_function
    >>> import numpy as np
    >>> from os.path import dirname, abspath, join as pjoin
    >>> import Corrfunc
    >>> from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
    >>> import math
    >>> binfile = pjoin(dirname(abspath(Corrfunc.__file__)),
    ...                 "../mocks/tests/", "bins")
    >>> N = 100000
    >>> boxsize = 420.0
    >>> seed = 42
    >>> np.random.seed(seed)
    >>> X = np.random.uniform(-0.5*boxsize, 0.5*boxsize, N)
    >>> Y = np.random.uniform(-0.5*boxsize, 0.5*boxsize, N)
    >>> Z = np.random.uniform(-0.5*boxsize, 0.5*boxsize, N)
    >>> weights = np.ones_like(X)
    >>> autocorr = 1
    >>> nthreads = 2
    >>> pimax = 40.0
    >>> results = DDrppi_mocks(autocorr, nthreads,
    ...                        pimax, binfile, X, Y, Z,
    ...                        weights1=weights, weight_type='pair_product',
    ...                        output_rpavg=True)
    >>> for r in results[519:]: print("{0:10.6f} {1:10.6f} {2:10.6f} {3:10.1f}"
    ...                               " {4:10d} {5:10.6f}".format(r['rmin'], r['rmax'],
    ...                               r['rpavg'], r['pimax'], r['npairs'], r['weightavg']))
    ...                         # doctest: +NORMALIZE_WHITESPACE
     11.359969  16.852277  14.285169       40.0     104850   1.000000
     16.852277  25.000000  21.181246        1.0     274144   1.000000
     16.852277  25.000000  21.190844        2.0     272876   1.000000
     16.852277  25.000000  21.183321        3.0     272294   1.000000
     16.852277  25.000000  21.188486        4.0     272506   1.000000
     16.852277  25.000000  21.170832        5.0     272100   1.000000
     16.852277  25.000000  21.165379        6.0     271788   1.000000
     16.852277  25.000000  21.175246        7.0     270040   1.000000
     16.852277  25.000000  21.187417        8.0     269492   1.000000
     16.852277  25.000000  21.172066        9.0     269682   1.000000
     16.852277  25.000000  21.182460       10.0     268266   1.000000
     16.852277  25.000000  21.170594       11.0     268744   1.000000
     16.852277  25.000000  21.178608       12.0     266820   1.000000
     16.852277  25.000000  21.187184       13.0     266510   1.000000
     16.852277  25.000000  21.184937       14.0     265484   1.000000
     16.852277  25.000000  21.180184       15.0     265258   1.000000
     16.852277  25.000000  21.191504       16.0     262952   1.000000
     16.852277  25.000000  21.187746       17.0     262602   1.000000
     16.852277  25.000000  21.189778       18.0     260206   1.000000
     16.852277  25.000000  21.188882       19.0     259410   1.000000
     16.852277  25.000000  21.185684       20.0     256806   1.000000
     16.852277  25.000000  21.194036       21.0     255574   1.000000
     16.852277  25.000000  21.184115       22.0     255406   1.000000
     16.852277  25.000000  21.178255       23.0     252394   1.000000
     16.852277  25.000000  21.184644       24.0     252220   1.000000
     16.852277  25.000000  21.187020       25.0     251668   1.000000
     16.852277  25.000000  21.183827       26.0     249648   1.000000
     16.852277  25.000000  21.183121       27.0     247160   1.000000
     16.852277  25.000000  21.180872       28.0     246238   1.000000
     16.852277  25.000000  21.185251       29.0     246030   1.000000
     16.852277  25.000000  21.183488       30.0     242124   1.000000
     16.852277  25.000000  21.194538       31.0     242426   1.000000
     16.852277  25.000000  21.190702       32.0     239778   1.000000
     16.852277  25.000000  21.188985       33.0     239046   1.000000
     16.852277  25.000000  21.187092       34.0     237640   1.000000
     16.852277  25.000000  21.185515       35.0     236256   1.000000
     16.852277  25.000000  21.190278       36.0     233536   1.000000
     16.852277  25.000000  21.183240       37.0     233274   1.000000
     16.852277  25.000000  21.183796       38.0     231628   1.000000
     16.852277  25.000000  21.200668       39.0     230378   1.000000
     16.852277  25.000000  21.181153       40.0     229006   1.000000

    """
    try:
        from Corrfunc._countpairs_mocks import countpairs_rp_pi_mocks as\
            DDrppi_extn
    except ImportError:
        msg = "Could not import the C extension for the on-sky"\
              "pair counter."
        raise ImportError(msg)

    import numpy as np
    from Corrfunc.utils import translate_isa_string_to_enum, translate_bin_type_string_to_enum, translate_los_type_string_to_enum,\
                               get_edges, convert_to_native_endian, sys_pipes, process_weights
    from future.utils import bytes_to_native_str

    if not autocorr:
        if X2 is None or Y2 is None or Z2 is None:
            msg = "Must pass valid arrays for X2/Y2/Z2 for "\
                  "computing cross-correlation"
            raise ValueError(msg)
    else:
        X2 = np.empty(1)
        Y2 = np.empty(1)
        Z2 = np.empty(1)

    weights1, weights2 = process_weights(weights1, weights2, X1, X2, weight_type, autocorr)

    # Ensure all input arrays are native endian
    X1, Y1, Z1, X2, Y2, Z2 = [
            convert_to_native_endian(arr, warn=False) for arr in
            [X1, Y1, Z1, X2, Y2, Z2]]

    if weights1 is not None:
        weights1 = [convert_to_native_endian(arr, warn=False) for arr in weights1]
    if weights2 is not None:
        weights2 = [convert_to_native_endian(arr, warn=False) for arr in weights2]

    if pair_weights is not None:
        pair_weights = convert_to_native_endian(pair_weights, warn=False)
        sep_pair_weights = convert_to_native_endian(sep_pair_weights, warn=False)

    # Passing None parameters breaks the parsing code, so avoid this
    kwargs = {}
    for k in ['weights1', 'weights2', 'weight_type', 'X2', 'Y2', 'Z2',
              'pair_weights', 'sep_pair_weights', 'attrs_pair_weights']:
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    integer_isa = translate_isa_string_to_enum(isa)
    integer_bin_type = translate_bin_type_string_to_enum(bin_type)
    integer_los_type = translate_los_type_string_to_enum(los_type)
    rbinfile = get_edges(binfile)
    with sys_pipes():
        extn_results = DDrppi_extn(autocorr, nthreads,
                                   rbinfile, pimax, npibins,
                                   X1, Y1, Z1,
                                   verbose=verbose,
                                   output_rpavg=output_rpavg,
                                   fast_divide_and_NR_steps=fast_divide_and_NR_steps,
                                   xbin_refine_factor=xbin_refine_factor,
                                   ybin_refine_factor=ybin_refine_factor,
                                   zbin_refine_factor=zbin_refine_factor,
                                   max_cells_per_dim=max_cells_per_dim,
                                   copy_particles=copy_particles,
                                   enable_min_sep_opt=enable_min_sep_opt,
                                   c_api_timer=c_api_timer,
                                   isa=integer_isa,
                                   bin_type=integer_bin_type,
                                   los_type=integer_los_type,
                                   **kwargs)
    if extn_results is None:
        msg = "RuntimeError occurred"
        raise RuntimeError(msg)
    else:
        extn_results, api_time = extn_results

    results_dtype = np.dtype([(bytes_to_native_str(b'rmin'), np.float64),
                              (bytes_to_native_str(b'rmax'), np.float64),
                              (bytes_to_native_str(b'rpavg'), np.float64),
                              (bytes_to_native_str(b'pimax'), np.float64),
                              (bytes_to_native_str(b'npairs'), np.uint64),
                              (bytes_to_native_str(b'weightavg'), np.float64)])
    results = np.array(extn_results, dtype=results_dtype)

    if not c_api_timer:
        return results
    else:
        return results, api_time


if __name__ == '__main__':
    import doctest
    doctest.testmod()
