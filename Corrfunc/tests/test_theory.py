#!/usr/bin/env python

from os.path import dirname, abspath, join as pjoin

import pytest
import numpy as np

from Corrfunc.tests.common import gals_Mr19
from Corrfunc.tests.common import (check_against_reference,
                                   check_vpf_against_reference)
from Corrfunc.tests.common import generate_isa_and_nthreads_combos



@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_DD(gals_Mr19, isa, nthreads):
    from Corrfunc.theory import DD

    boxsize = 420.
    binfile = pjoin(dirname(abspath(__file__)),
                     "../../theory/tests/", "bins")
    autocorr = 1
    periodic = 1

    x, y, z, w = gals_Mr19
    for size in [0, None]:
        results_DD = DD(autocorr, nthreads, binfile,
                        x[:size], y[:size], z[:size],
                        weights1=w[:size], weight_type='pair_product',
                        periodic=periodic, boxsize=boxsize,
                        output_ravg=True, verbose=True,
                        isa=isa)
        if size == 0:
            for name in ['npairs', 'weightavg', 'ravg']: assert np.allclose(results_DD[name], 0.)
        if size is None:
            file_ref = pjoin(dirname(abspath(__file__)),
                            "../../theory/tests/", "Mr19_DD_periodic")
            check_against_reference(results_DD, file_ref,
                                    ravg_name='ravg', ref_cols=(0, 4, 1))


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_DDrppi(gals_Mr19, isa, nthreads):
    from Corrfunc.theory import DDrppi

    boxsize = 420.
    pimax = 40.0
    binfile = pjoin(dirname(abspath(__file__)),
                     "../../theory/tests/", "bins")
    autocorr = 1
    periodic = 1

    x, y, z, w = gals_Mr19
    for size in [0, None]:
        results_DDrppi = DDrppi(autocorr, nthreads, binfile, pimax, int(pimax),
                                x[:size], y[:size], z[:size],
                                weights1=w[:size], weight_type='pair_product',
                                periodic=periodic, boxsize=boxsize,
                                output_rpavg=True, verbose=True,
                                isa=isa)

    if size == 0:
        for name in ['npairs', 'weightavg', 'rpavg']: assert np.allclose(results_DDrppi[name], 0.)
    if size is None:
        file_ref = pjoin(dirname(abspath(__file__)),
                        "../../theory/tests/", "Mr19_DDrppi_periodic")
        check_against_reference(results_DDrppi, file_ref, ravg_name='rpavg', ref_cols=(0, 4, 1))


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_DDsmu(gals_Mr19, isa, nthreads):
    from Corrfunc.theory import DDsmu

    boxsize = 420.
    binfile = pjoin(dirname(abspath(__file__)),
                     "../../theory/tests/", "bins")
    autocorr = 1
    periodic = 1
    mu_max = 0.5
    nmu_bins = 11

    x, y, z, w = gals_Mr19
    for size in [0, None]:
        results_DDsmu = DDsmu(autocorr, nthreads, binfile,
                              mu_max, nmu_bins,
                              x[:size], y[:size], z[:size],
                              weights1=w[:size], weight_type='pair_product',
                              periodic=periodic, boxsize=boxsize,
                              output_savg=True, verbose=True,
                              isa=isa)

    if size == 0:
        for name in ['npairs', 'weightavg', 'savg']: assert np.allclose(results_DDsmu[name], 0.)
    if size is None:
        file_ref = pjoin(dirname(abspath(__file__)),
                        "../../theory/tests/", "Mr19_DDsmu_periodic")
        check_against_reference(results_DDsmu, file_ref, ravg_name='savg', ref_cols=(0, 4, 1))


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos(extra_isa=['AVX2']))
def test_wp(gals_Mr19, isa, nthreads):
    from Corrfunc.theory import wp

    boxsize = 420.
    pimax = 40.
    binfile = pjoin(dirname(abspath(__file__)),
                     "../../theory/tests/", "bins")

    x, y, z, w = gals_Mr19
    for size in [0, None]:
        results_wp = wp(boxsize, nthreads, binfile, pimax,
                        x[:size], y[:size], z[:size],
                        weights=w[:size], weight_type='pair_product',
                        output_rpavg=True, verbose=True,
                        isa=isa)
        if size == 0:
            for name in ['npairs', 'weightavg', 'rpavg']: assert np.allclose(results_wp[name], 0.)
        if size is None:
            file_ref = pjoin(dirname(abspath(__file__)),
                            "../../theory/tests/", "Mr19_wp")
            check_against_reference(results_wp, file_ref, ravg_name='rpavg', cf_name='wp',
                                    ref_cols=(4, 5, 1, 0))


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_xi(gals_Mr19, isa, nthreads):
    from Corrfunc.theory import xi

    boxsize = 420.
    binfile = pjoin(dirname(abspath(__file__)),
                     "../../theory/tests/", "bins")

    x, y, z, w = gals_Mr19
    for size in [0, None]:
        results_xi = xi(boxsize, nthreads, binfile,
                        x[:size], y[:size], z[:size],
                        weights=w[:size], weight_type='pair_product',
                        output_ravg=True, verbose=True,
                        isa=isa)

    if size == 0:
        for name in ['npairs', 'weightavg', 'rpavg']: assert np.allclose(results_xi[name], 0.)
    if size is None:
        file_ref = pjoin(dirname(abspath(__file__)),
                        "../../theory/tests/", "Mr19_xi")
        check_against_reference(results_xi, file_ref, ravg_name='ravg', cf_name='xi', ref_cols=(4, 5, 1, 0))


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_vpf(gals_Mr19, isa, nthreads):
    from Corrfunc.theory import vpf

    boxsize = 420.
    rmax = 10.0
    nbin = 10
    nspheres = 10000
    num_pN = 6
    seed = -1234
    periodic = 1

    x,y,z,w = gals_Mr19
    results_vpf  = vpf(rmax, nbin, nspheres, num_pN,
                       seed, x, y, z, verbose=True, periodic=periodic,
                       boxsize=boxsize)
    #results_vpf = results_vpf.view(dtype=np.float64).reshape(nbin,-1)  # flatten to same shape as results
    file_ref = pjoin(dirname(abspath(__file__)),
                    "../../theory/tests/", "Mr19_vpf_periodic")
    check_vpf_against_reference(results_vpf, file_ref)
