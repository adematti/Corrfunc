#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python wrapper around the C extension for the pair counter in
``mocks/DDbessel``. This python wrapper is :py:mod:`Corrfunc.mocks.DDbessel_mocks`
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__author__ = ('Arnaud de Mattia')
__all__ = ('DDbessel_mocks', )


def DDbessel_mocks(autocorr, nthreads, binfile, ells, rmin, rmax, mumax,
                   X1, Y1, Z1, XP1=None, YP1=None, ZP1=None, weights1=None,
                   X2=None, Y2=None, Z2=None, XP2=None, YP2=None, ZP2=None, weights2=None,
                   verbose=False,
                   xbin_refine_factor=2, ybin_refine_factor=2,
                   zbin_refine_factor=1, max_cells_per_dim=100,
                   copy_particles=True, enable_min_sep_opt=True,
                   c_api_timer=False, isa='fastest',
                   weight_type=None, los_type='midpoint',
                   pair_weights=None, sep_pair_weights=None, attrs_pair_weights=None):
    """
    Calculate the 2-D pair-counts corresponding to the projected correlation
    function, :math:`\\xi(s, \\mu)`. The pairs are counted in bins of
    radial separation and cosine of angle to the line-of-sight (LOS). The
    input positions are expected to be on-sky co-ordinates. This module is
    suitable for calculating correlation functions for mock catalogs.

    If ``weights`` are provided, the resulting pair counts are weighted.  The
    weighting scheme depends on ``weight_type``.

    Returns a numpy structured array containing the pair counts for the
    specified bins.


    .. note:: This module only returns pair counts and not the actual
       correlation function :math:`\\xi(s, \\mu)`. See the
       utilities :py:mod:`Corrfunc.utils.convert_3d_counts_to_cf`
       for computing :math:`\\xi(s, \\mu)` from the pair counts.

    .. versionadded:: 2.4.0

    Parameters
    ----------
    autocorr : boolean, required
        Boolean flag for auto/cross-correlation. If autocorr is set to 1,
        then the second set of particle positions are not required.

    nthreads : integer
        The number of OpenMP threads to use. Has no effect if OpenMP was not
        enabled during library compilation.

    binfile : string or an list/array of floats
        For string input: filename specifying the ``r`` bins for
        ``DD``. The file should contain white-space separated values
        of (rmin, rmax)  for each ``r`` wanted. The bins need to be
        contiguous and sorted in increasing order (smallest bins come first).

        For array-like input: A sequence of ``r`` values that provides the
        bin-edges. For example,
        ``np.logspace(np.log10(0.1), np.log10(10.0), 15)`` is a valid
        input specifying **14** (logarithmic) bins between 0.1 and 10.0. This
        array does not need to be sorted.

    X1/Y1/Z1 : array_like, real (float/double)
        The array of X/Y/Z positions for the first set of points.
        Calculations are done in the precision of the supplied arrays.

    weights1 : array_like, real (float/double), optional
        A scalar, or an array of weights of shape (n_weights, n_positions) or
        (n_positions,). ``weight_type`` specifies how these weights are used;
        results are returned in the ``weightavg`` field.  If only one of
        weights1 and weights2 is specified, the other will be set to uniform
        weights.

    X2/Y2/Z2 : array-like, real (float/double)
        Array of XYZ positions for the second set of points. *Must* be the same
        precision as the X1/Y1/Z1 arrays. Only required when ``autocorr==0``.

    weights2 : array-like, real (float/double), optional
        Same as weights1, but for the second set of positions

    verbose : boolean (default false)
        Boolean flag to control output of informational messages

    output_ravg : boolean (default false)
        Boolean flag to output the average ``r`` for each bin. Code will
        run slower if you set this flag.

        Note: If you are calculating in single-precision, ``ravg`` will
        suffer from numerical loss of precision and can not be trusted.
        If you need accurate ``ravg`` values, then pass in double precision
        arrays for the particle positions.

    (xyz)bin_refine_factor : integer, default is (2,2,1); typically within [1-3]
        Controls the refinement on the cell sizes. Can have up to a 20% impact
        on runtime.

    max_cells_per_dim : integer, default is 100, typical values in [50-300]
        Controls the maximum number of cells per dimension. Total number of
        cells can be up to (max_cells_per_dim)^3. Only increase if ``rmax`` is
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

    isa : string (default ``fastest``)
        Controls the runtime dispatch for the instruction set to use. Options
        are: [``fastest``, ``avx512f``, ``avx``, ``sse42``, ``fallback``]

        Setting isa to ``fastest`` will pick the fastest available instruction
        set on the current computer. However, if you set ``isa`` to, say,
        ``avx`` and ``avx`` is not available on the computer, then the code
        will revert to using ``fallback`` (even though ``sse42`` might be
        available).  Unless you are benchmarking the different instruction
        sets, you should always leave ``isa`` to the default value. And if
        you *are* benchmarking, then the string supplied here gets translated
        into an ``enum`` for the instruction set defined in ``utils/defs.h``.

    weight_type : string, optional. Default: None.
        The type of weighting to apply. One of ["pair_product", "inverse_bitwise", None].

    bin_type : string, case-insensitive (default ``custom``)
        Set to ``lin`` for speed-up in case of linearly-spaced bins.
        In this case, the bin number for a pair separated by ``r`` is given by
        ``(r - binfile[0])/(binfile[-1] - binfile[0])*(len(binfile) - 1)``,
        i.e. only the first and last bins of input ``binfile`` are considered.
        Then setting ``output_ravg`` is virtually costless.
        For non-linear binning, set to ``custom``.
        In the vast majority of cases, bin_type='linear' will yield identical
        results to custom linear binning but with higher performance.
        In a few rare cases where a pair falls on a bin boundary,
        'linear' and custom linear may disagree on which bin the pair falls into
        due to finite floating point precision.
        ``auto`` will choose linear binning if input ``binfile`` is within
        ``rtol = 1e-05`` *and* ``atol = 1e-08`` (relative and absolute tolerance)
        of ``np.linspace(binfile[0], binfile[-1], len(binfile))``.

    pair_weights : array-like, optional. Default: None.
        Array of pair weights.

    sep_pair_weights : array-like, optional. Default: None.
        Array of separations corresponding to ``pair_weights``.

    attrs_pair_weights : tuple. Default: None.
        Attributes for pair weights; in case ``weight_type`` is "inverse_bitwise",
        the tuple of (offset to be added to the bitwise counts,
        default weight value if denominator is zero).

    Returns
    -------
    results : Numpy structured array
        A numpy structured array containing [smin, smax, savg, mumax,
        npairs, weightavg]. There are a total of ``nmubins`` in ``mu``
        for each separation bin specified in the ``binfile``, with ``mumax``
        being the upper limit of the ``mu`` bin. If ``output_savg`` is  not
        set, then ``savg`` will be set to 0.0 for all bins; similarly for
        ``weightavg``. ``npairs`` contains the number of pairs in that bin
        and can be used to compute the actual :math:`\\xi(s, \\mu)` by
        combining with (DR, RR) counts.

    api_time : float, optional
        Only returned if ``c_api_timer`` is set.  ``api_time`` measures only
        the time spent within the C library and ignores all python overhead.
    """
    try:
        from Corrfunc._countpairs_mocks import countpairs_bessel_mocks as DDbessel_extn
    except ImportError as exc:
        msg = "Could not import the C extension for bessel"\
              "pair counter."
        raise ImportError(msg) from exc

    import numpy as np
    from Corrfunc.utils import translate_isa_string_to_enum, translate_bin_type_string_to_enum,\
                               get_edges, convert_to_native_endian,\
                               sys_pipes, process_weights
    from future.utils import bytes_to_native_str

    # Check if mumax is scalar
    if np.ndim(mumax) != 0:
        msg = "The parameter `mumax` = {0}, has size = {1}. "\
              "The code is expecting a scalar quantity (and not "\
              "not a list, array)".format(mumax, np.size(mumax))
        raise TypeError(msg)

    # Check that mumax is within (0.0, 1.0]
    if mumax <= 0.: #or mumax > 1.0:
        msg = "The parameter `mumax` = {0}, is the max. of cosine of an "\
        "angle and should be within (0.0, 1.0]".format(mumax)
        raise ValueError(msg)

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
    X1, Y1, Z1, X2, Y2, Z2, XP1, YP1, ZP1, XP2, YP2, ZP2 = [
                        convert_to_native_endian(arr, warn=True) for arr in
                        [X1, Y1, Z1, X2, Y2, Z2, XP1, YP1, ZP1, XP2, YP2, ZP2]]
    XP1, YP1, ZP1, XP2, YP2, ZP2 = [xp if xp is not None else x for x, xp in
                        [(XP1, X1), (YP1, Y1), (ZP1, Z1), (XP2, X2), (YP2, Y2), (ZP2, Z2)]]

    if weights1 is None:
        weights1 = []
    else:
        weights1 = [convert_to_native_endian(arr, warn=True) for arr in weights1]
    weights1 += [XP1, YP1, ZP1]

    if weights2 is None:
        weights2 = []
    else:
        weights2 = [convert_to_native_endian(arr, warn=True) for arr in weights2]
    weights2 += [XP2, YP2, ZP2]

    if pair_weights is not None:
        pair_weights = convert_to_native_endian(pair_weights, warn=True)
        sep_pair_weights = convert_to_native_endian(sep_pair_weights, warn=True)

    # Passing None parameters breaks the parsing code, so avoid this
    kwargs = {}
    for k in ['weights1', 'weights2', 'weight_type', 'X2', 'Y2', 'Z2',
              'pair_weights', 'sep_pair_weights', 'attrs_pair_weights']:
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    ells = np.asarray(ells, dtype='i4').ravel()
    integer_isa = translate_isa_string_to_enum(isa)
    integer_los_type = {'midpoint':0, 'firstpoint':1}[los_type.lower()]
    binfile = get_edges(binfile)
    with sys_pipes():
        extn_results = DDbessel_extn(autocorr, nthreads,
                                     binfile, ells, rmin, rmax, mumax,
                                     X1, X1, Z1,
                                     verbose=verbose,
                                     xbin_refine_factor=xbin_refine_factor,
                                     ybin_refine_factor=ybin_refine_factor,
                                     zbin_refine_factor=zbin_refine_factor,
                                     max_cells_per_dim=max_cells_per_dim,
                                     copy_particles=copy_particles,
                                     enable_min_sep_opt=enable_min_sep_opt,
                                     c_api_timer=c_api_timer,
                                     isa=integer_isa,
                                     los_type=integer_los_type,
                                     **kwargs)
    if extn_results is None:
        msg = "RuntimeError occurred"
        raise RuntimeError(msg)
    else:
        extn_results, api_time = extn_results

    results_dtype = np.dtype([(bytes_to_native_str(b'ells'), np.int32),
                              (bytes_to_native_str(b'modes'), np.float64),
                              (bytes_to_native_str(b'poles'), np.float64)])

    nbin = len(extn_results)
    results = np.zeros(nbin, dtype=results_dtype)
    for ii, r in enumerate(extn_results):
        results['ells'][ii] = r[0]
        results['modes'][ii] = r[1]
        results['poles'][ii] = r[2]

    if not c_api_timer:
        return results
    return results, api_time


if __name__ == '__main__':
    import doctest
    doctest.testmod()
