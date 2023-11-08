#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python wrapper around the C extension for the pair counter in
``mocks/DDsmu``. This python wrapper is :py:mod:`Corrfunc.mocks.DDsmu_mocks`
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import warnings

__author__ = ('Manodeep Sinha', 'Nick Hand')
__all__ = ('DDsmu_mocks', )


def DDsmu_mocks(autocorr, nthreads, binfile, mumax, nmubins,
                X1, Y1, Z1, weights1=None,
                X2=None, Y2=None, Z2=None, weights2=None,
                verbose=False, output_savg=False,
                fast_divide_and_NR_steps=0,
                xbin_refine_factor=2, ybin_refine_factor=2,
                zbin_refine_factor=1, max_cells_per_dim=100,
                copy_particles=True, enable_min_sep_opt=True,
                c_api_timer=False, isa='fastest', gpu=False,
                weight_type=None, bin_type='custom', los_type='midpoint',
                pair_weights=None, sep_pair_weights=None, attrs_pair_weights=None, attrs_selection=None):
    """
    Calculate the 2-D pair-counts corresponding to the correlation
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

    .. versionadded:: 2.1.0

    Parameters
    ----------
    autocorr : boolean, required
        Boolean flag for auto/cross-correlation. If autocorr is set to 1,
        then the second set of particle positions are not required.

    nthreads : integer
        The number of OpenMP threads to use. Has no effect if OpenMP was not
        enabled during library compilation.

    binfile : string or an list/array of floats
        For string input: filename specifying the ``s`` bins for
        ``DDsmu_mocks``. The file should contain white-space separated values
        of (smin, smax) specifying each ``s`` bin wanted. The bins
        need to be contiguous and sorted in increasing order (smallest bins
        come first).

        For array-like input: A sequence of ``s`` values that provides the
        bin-edges. For example,
        ``np.logspace(np.log10(0.1), np.log10(10.0), 15)`` is a valid
        input specifying **14** (logarithmic) bins between 0.1 and 10.0. This
        array does not need to be sorted.

    mumax : double. Must be in range [0.0, 1.0]
        A double-precision value for the maximum cosine of the angular
        separation from the line of sight (LOS). Here, ``mu`` is defined as
        the angle between ``s`` and ``l``.

        Note: pairs with :math:`-\\mu_{max} < \\mu < \\mu_{max}`
        (exclusive on both ends) are counted.

    nmubins : int
        The number of linear ``mu`` bins, with the bins ranging from
        from (:math:`-\\mu_{max}`, :math:`\\mu_{max}`).

    X1/Y1/Z1 : array_like, real (float/double)
        The array of X/Y/Z positions for the first set of points.
        Calculations are done in the precision of the supplied arrays.

    weights1 : array_like, real (float/double), optional
        A scalar, or an array of weights of shape (n_weights, n_positions)
        or (n_positions,). `weight_type` specifies how these weights are used;
        results are returned in the `weightavg` field.  If only one of
        ``weights1`` or ``weights2`` is specified, the other will be set
        to uniform weights.

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

    output_savg : boolean (default false)
        Boolean flag to output the average ``s`` for each bin. Code will
        run slower if you set this flag. Also, note, if you are calculating
        in single-precision, ``savg`` will suffer from numerical loss of
        precision and can not be trusted. If you need accurate ``savg``
        values, then pass in double precision arrays for the particle
        positions.

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

    isa : string, case-insensitive (default ``fastest``)
       Controls the runtime dispatch for the instruction set to use. Options
       are: [``fastest``, ``avx512f``, ``avx``, ``sse42``, ``fallback``]

       Setting isa to ``fastest`` will pick the fastest available instruction
       set on the current computer. However, if you set ``isa`` to, say,
       ``avx`` and ``avx`` is not available on the computer, then the code will
       revert to using ``fallback`` (even though ``sse42`` might be available).

       Unless you are benchmarking the different instruction sets, you should
       always leave ``isa`` to the default value. And if you *are*
       benchmarking, then the string supplied here gets translated into an
       ``enum`` for the instruction set defined in ``utils/defs.h``.

    gpu : bool (default False)
        If ``True``, use GPU (nvidia) instead of CPU.

    weight_type : string, optional (default None)
        The type of weighting to apply. One of ["pair_product", "inverse_bitwise", None].

    bin_type : string, case-insensitive (default ``custom``)
        Set to ``lin`` for speed-up in case of linearly-spaced bins.
        In this case, the bin number for a pair separated by ``s`` is given by
        ``(s - binfile[0])/(binfile[-1] - binfile[0])*(len(binfile) - 1)``,
        i.e. only the first and last bins of input ``binfile`` are considered.
        Then setting ``output_savg`` is virtually costless.
        For non-linear binning, set to ``custom``.
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

    attrs_pair_weights : tuple. Default: None.
        Attributes for pair weights; in case ``weight_type`` is "inverse_bitwise",
        the tuple of (offset to be added to the bitwise counts,
        default weight value if denominator is zero).

    attrs_selection : dict. Default=None.
        To select pairs to be counted, provide mapping between the quantity (str)
        and the interval (tuple of floats),
        e.g. ``{'rp': (0., 20.)}`` to select pairs with transverse separation 'rp' between 0 and 20,
        `{'theta': (0., 20.)}`` to select pairs with separation angle 'theta' between 0 and 20 degrees.

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
        from Corrfunc._countpairs_mocks import countpairs_s_mu_mocks as\
            DDsmu_extn
    except ImportError:
        msg = "Could not import the C extension for the on-sky"\
              "pair counter."
        raise ImportError(msg)

    import numpy as np
    from Corrfunc.utils import translate_isa_string_to_enum, translate_bin_type_string_to_enum, translate_los_type_string_to_enum,\
                               get_edges, convert_to_native_endian, sys_pipes, process_weights
    from future.utils import bytes_to_native_str

    # Check if mumax is scalar
    if not np.isscalar(mumax):
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
              'pair_weights', 'sep_pair_weights', 'attrs_pair_weights', 'attrs_selection']:
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    integer_isa = translate_isa_string_to_enum(isa)
    integer_bin_type = translate_bin_type_string_to_enum(bin_type)
    integer_los_type = translate_los_type_string_to_enum(los_type)
    sbinfile = get_edges(binfile)

    warn_large_mu(mumax,
                  # X1, Y1, Z1 are checked to be the same dtype
                  dtype=X1.dtype,
                  )

    with sys_pipes():
        extn_results = DDsmu_extn(autocorr, nthreads,
                                  sbinfile, mumax, nmubins,
                                  X1, Y1, Z1,
                                  verbose=verbose,
                                  output_savg=output_savg,
                                  fast_divide_and_NR_steps=fast_divide_and_NR_steps,
                                  xbin_refine_factor=xbin_refine_factor,
                                  ybin_refine_factor=ybin_refine_factor,
                                  zbin_refine_factor=zbin_refine_factor,
                                  max_cells_per_dim=max_cells_per_dim,
                                  copy_particles=copy_particles,
                                  enable_min_sep_opt=enable_min_sep_opt,
                                  c_api_timer=c_api_timer,
                                  isa=integer_isa,
                                  gpu=int(gpu),
                                  bin_type=integer_bin_type,
                                  los_type=integer_los_type,
                                  **kwargs)
    if extn_results is None:
        msg = "RuntimeError occurred"
        raise RuntimeError(msg)
    else:
        extn_results, api_time = extn_results

    results_dtype = np.dtype([(bytes_to_native_str(b'smin'), np.float64),
                              (bytes_to_native_str(b'smax'), np.float64),
                              (bytes_to_native_str(b'savg'), np.float64),
                              (bytes_to_native_str(b'mumax'), np.float64),
                              (bytes_to_native_str(b'npairs'), np.uint64),
                              (bytes_to_native_str(b'weightavg'), np.float64)])

    nbin = len(extn_results)
    results = np.zeros(nbin, dtype=results_dtype)
    for ii, r in enumerate(extn_results):
        results['smin'][ii] = r[0]
        results['smax'][ii] = r[1]
        results['savg'][ii] = r[2]
        results['mumax'][ii] = r[3]
        results['npairs'][ii] = r[4]
        results['weightavg'][ii] = r[5]

    if not c_api_timer:
        return results
    else:
        return results, api_time


def warn_large_mu(mu_max, dtype):
    '''
    Small theta values (large mu) underfloat float32. Warn the user.
    Context: https://github.com/manodeep/Corrfunc/issues/296 (see also #297)
    '''
    if dtype.itemsize > 4:
        return

    if mu_max >= 0.9800666:  # cos(0.2)
        warnings.warn("""
Be aware that small angular pair separations (mu near 1) will suffer from loss
of floating-point precision, as the input data is in float32 precision or
lower. In float32, the loss of precision is 1% in mu at separations of 0.2
degrees, and larger at smaller separations.
For more information, see:
https://github.com/manodeep/Corrfunc/issues/296 (see also #297)
"""
                      )


if __name__ == '__main__':
    import doctest
    doctest.testmod()
