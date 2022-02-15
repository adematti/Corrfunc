#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python wrapper around the C extension for the pair counter in
``mocks/DDsmu``. This python wrapper is :py:mod:`Corrfunc.mocks.DDsmu_mocks`
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__author__ = ('Manodeep Sinha', 'Nick Hand')
__all__ = ('DDsmu_mocks', )


def DDsmu_mocks(autocorr, cosmology, nthreads, binfile, mumax, nmubins,
                RA1, DEC1, CZ1, weights1=None,
                RA2=None, DEC2=None, CZ2=None, weights2=None,
                is_comoving_dist=False,
                verbose=False, output_savg=False,
                fast_divide_and_NR_steps=0,
                xbin_refine_factor=2, ybin_refine_factor=2,
                zbin_refine_factor=1, max_cells_per_dim=100,
                copy_particles=True, enable_min_sep_opt=True,
                c_api_timer=False, isa='fastest',
                weight_type=None, bin_type='custom',
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

    .. versionadded:: 2.1.0

    Parameters
    ----------
    autocorr : boolean, required
        Boolean flag for auto/cross-correlation. If autocorr is set to 1,
        then the second set of particle positions are not required.

    cosmology : integer, required
        Integer choice for setting cosmology. Valid values are 1->LasDamas
        cosmology and 2->Planck cosmology. If you need arbitrary cosmology,
        easiest way is to convert the ``CZ`` values into co-moving distance,
        based on your preferred cosmology. Set ``is_comoving_dist=True``, to
        indicate that the co-moving distance conversion has already been done.

        Choices:
                 1. LasDamas cosmology. :math:`\\Omega_m=0.25`, :math:`\\Omega_\Lambda=0.75`
                 2. Planck   cosmology. :math:`\\Omega_m=0.302`, :math:`\\Omega_\Lambda=0.698`

        To setup a new cosmology, add an entry to the function,
        ``init_cosmology`` in ``ROOT/utils/cosmology_params.c`` and re-install
        the entire package.

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
        the angle between ``s`` and ``l``. If :math:`v_1` and :math:`v_2`
        represent the vectors to each point constituting the pair, then
        :math:`s := v_1 - v_2` and :math:`l := 1/2 (v_1 + v_2)`.

        Note: Pairs with :math:`-\\mu_{max} < \\mu < \\mu_{max}`
        (exclusive on both ends) are counted.

    nmubins : int
        The number of linear ``mu`` bins, with the bins ranging from
        from (:math:`-\\mu_{max}`, :math:`\\mu_{max}`).

    RA1 : array-like, real (float/double)
        The array of Right Ascensions for the first set of points. RA's
        are expected to be in [0.0, 360.0], but the code will try to fix cases
        where the RA's are in [-180, 180.0]. For peace of mind, always supply
        RA's in [0.0, 360.0].

        Calculations are done in the precision of the supplied arrays.

    DEC1 : array-like, real (float/double)
        Array of Declinations for the first set of points. DEC's are expected
        to be in the [-90.0, 90.0], but the code will try to fix cases where
        the DEC's are in [0.0, 180.0]. Again, for peace of mind, always supply
        DEC's in [-90.0, 90.0].

        Must be of same precision type as RA1.

    CZ1 : array-like, real (float/double)
        Array of (Speed Of Light * Redshift) values for the first set of
        points. Code will try to detect cases where ``redshifts`` have been
        passed and multiply the entire array with the ``speed of light``.

        If is_comoving_dist is set, then ``CZ1`` is interpreted as the
        co-moving distance, rather than `cz`.

    weights1 : array_like, real (float/double), optional
        A scalar, or an array of weights of shape (n_weights, n_positions)
        or (n_positions,). `weight_type` specifies how these weights are used;
        results are returned in the `weightavg` field.  If only one of
        ``weights1`` or ``weights2`` is specified, the other will be set
        to uniform weights.

    RA2 : array-like, real (float/double)
        The array of Right Ascensions for the second set of points. RA's
        are expected to be in [0.0, 360.0], but the code will try to fix cases
        where the RA's are in [-180, 180.0]. For peace of mind, always supply
        RA's in [0.0, 360.0].

        Must be of same precision type as RA1/DEC1/CZ1.

    DEC2 : array-like, real (float/double)
        Array of Declinations for the second set of points. DEC's are expected
        to be in the [-90.0, 90.0], but the code will try to fix cases where
        the DEC's are in [0.0, 180.0]. Again, for peace of mind, always supply
        DEC's in [-90.0, 90.0].

        Must be of same precision type as RA1/DEC1/CZ1.

    CZ2 : array-like, real (float/double)
        Array of (Speed Of Light * Redshift) values for the second set of
        points. Code will try to detect cases where ``redshifts`` have been
        passed and multiply the entire array with the ``speed of light``.

        If is_comoving_dist is set, then ``CZ2`` is interpreted as the
        co-moving distance, rather than `cz`.

        Must be of same precision type as RA1/DEC1/CZ1.

    weights2 : array-like, real (float/double), optional
        Same as weights1, but for the second set of positions

    is_comoving_dist : boolean (default false)
        Boolean flag to indicate that ``cz`` values have already been
        converted into co-moving distances. This flag allows arbitrary
        cosmologies to be used in ``Corrfunc``.

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

    weight_type : string, optional (default None)
        The type of weighting to apply.  One of ["pair_product", None].

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
        from Corrfunc._countpairs_mocks import countpairs_s_mu_mocks as\
            DDsmu_extn
    except ImportError:
        msg = "Could not import the C extension for the on-sky"\
              "pair counter."
        raise ImportError(msg)

    import numpy as np
    from Corrfunc.utils import translate_isa_string_to_enum, translate_bin_type_string_to_enum,\
        fix_ra_dec, get_edges, convert_to_native_endian,\
        sys_pipes, process_weights
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
        if RA2 is None or DEC2 is None or CZ2 is None:
            msg = "Must pass valid arrays for RA2/DEC2/CZ2 for "\
                  "computing cross-correlation"
            raise ValueError(msg)
    else:
        RA2 = np.empty(1)
        DEC2 = np.empty(1)
        CZ2 = np.empty(1)

    weights1, weights2 = process_weights(weights1, weights2, RA1, RA2, weight_type, autocorr)

    # Ensure all input arrays are native endian
    RA1, DEC1, CZ1, RA2, DEC2, CZ2 = [
            convert_to_native_endian(arr, warn=True) for arr in
            [RA1, DEC1, CZ1, RA2, DEC2, CZ2]]


    fix_ra_dec(RA1, DEC1)
    if autocorr == 0:
        fix_ra_dec(RA2, DEC2)

    if weights1 is not None:
        weights1 = [convert_to_native_endian(arr, warn=True) for arr in weights1]
    if weights2 is not None:
        weights2 = [convert_to_native_endian(arr, warn=True) for arr in weights2]

    if pair_weights is not None:
        pair_weights = convert_to_native_endian(pair_weights, warn=True)
        sep_pair_weights = convert_to_native_endian(sep_pair_weights, warn=True)

    # Passing None parameters breaks the parsing code, so avoid this
    kwargs = {}
    for k in ['weights1', 'weights2', 'weight_type', 'RA2', 'DEC2', 'CZ2',
              'pair_weights', 'sep_pair_weights', 'attrs_pair_weights']:
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    integer_isa = translate_isa_string_to_enum(isa)
    integer_bin_type = translate_bin_type_string_to_enum(bin_type)
    sbinfile = get_edges(binfile)
    with sys_pipes():
        extn_results = DDsmu_extn(autocorr, cosmology, nthreads,
                                  sbinfile, mumax, nmubins,
                                  RA1, DEC1, CZ1,
                                  is_comoving_dist=is_comoving_dist,
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
                                  bin_type=integer_bin_type, **kwargs)
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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
