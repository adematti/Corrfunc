#!/usr/bin/env python

from os.path import dirname, abspath, join as pjoin

import pytest
import numpy as np

from Corrfunc.tests.common import gals_Mr19, Mr19_mock_northonly, Mr19_mock_northonly_rdz, Mr19_randoms_northonly
from Corrfunc.tests.common import check_against_reference, check_vpf_against_reference
from Corrfunc.tests.common import generate_isa_and_nthreads_combos


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_DDbessel_mocks(gals_Mr19, isa, nthreads):
    from Corrfunc.mocks import DDbessel_mocks

    binfile = np.linspace(0.1, 0.3, 21)
    autocorr = 1
    ells = (0, 2, 4)

    x, y, z, w = gals_Mr19
    for size in [0, 100]:
        results_DDbessel_mocks = DDbessel_mocks(autocorr, nthreads,
                                                binfile, ells,
                                                0., 1., 1.,
                                                x[:size], y[:size], z[:size], weights1=w[:size],
                                                weight_type='pair_product',
                                                verbose=True,
                                                isa=isa)
        if size == 0:
            for name in ['poles']: assert np.allclose(results_DDbessel_mocks[name], 0.)



@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_DDrppi_mocks(Mr19_mock_northonly, isa, nthreads):
    from Corrfunc.mocks import DDrppi_mocks

    pimax = 40.0
    binfile = pjoin(dirname(abspath(__file__)),
                     "../../mocks/tests/", "bins")
    autocorr = 1

    x, y, z, w = Mr19_mock_northonly
    for size in [0, None]:
        results_DDrppi_mocks = DDrppi_mocks(autocorr, nthreads,
                                            binfile, pimax, int(pimax),
                                            x[:size], y[:size], z[:size], weights1=w[:size],
                                            weight_type='pair_product',
                                            output_rpavg=True, verbose=True,
                                            isa=isa)
        if size == 0:
            for name in ['npairs', 'weightavg', 'rpavg']: assert np.allclose(results_DDrppi_mocks[name], 0.)
        if size is None:
            file_ref = pjoin(dirname(abspath(__file__)),
                            "../../mocks/tests/", "Mr19_mock.DD")
            check_against_reference(results_DDrppi_mocks, file_ref, ravg_name='rpavg', ref_cols=(0, 4, 1))


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_DDsmu_mocks(Mr19_randoms_northonly, isa, nthreads):
    from Corrfunc.mocks import DDsmu_mocks

    binfile = pjoin(dirname(abspath(__file__)),
                     "../../mocks/tests/", "bins")
    autocorr = 1
    mu_max = 1.0
    nmu_bins = 11

    x, y, z, w = Mr19_randoms_northonly
    for size in [0, None]:
        results_DDsmu_mocks = DDsmu_mocks(autocorr, nthreads,
                                          binfile, mu_max, nmu_bins,
                                          x[:size], y[:size], z[:size], weights1=w[:size],
                                          weight_type='pair_product',
                                          output_savg=True, verbose=True,
                                          isa=isa)
        if size == 0:
            for name in ['npairs', 'weightavg', 'savg']: assert np.allclose(results_DDsmu_mocks[name], 0.)
        if size is None:
            file_ref = pjoin(dirname(abspath(__file__)),
                            "../../mocks/tests/", "Mr19_mock_DDsmu.RR")
            check_against_reference(results_DDsmu_mocks, file_ref, ravg_name='savg', ref_cols=(0, 4, 1))


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_DDtheta_mocks(Mr19_mock_northonly_rdz, isa, nthreads):
    from Corrfunc.mocks import DDtheta_mocks

    autocorr = 1
    binfile = pjoin(dirname(abspath(__file__)),
                     "../../mocks/tests/", "angular_bins")

    ra, dec, cz, w = Mr19_mock_northonly_rdz
    for size in [0, None]:
        for offset in [(0., 0.), (380., 180.)]:
            results_DDtheta_mocks = DDtheta_mocks(autocorr, nthreads, binfile,
                                                  ra[:size] + offset[0], dec[:size] + offset[1], weights1=w[:size],
                                                  weight_type='pair_product',
                                                  output_thetaavg=True, fast_acos=False,
                                                  verbose=True, isa=isa)
        if size == 0:
            for name in ['npairs', 'weightavg', 'thetaavg']:
                assert np.allclose(results_DDtheta_mocks[name], 0.)
        if size is None:
            file_ref = pjoin(dirname(abspath(__file__)),
                            "../../mocks/tests/", "Mr19_mock_wtheta.DD")
            check_against_reference(results_DDtheta_mocks, file_ref, ravg_name='thetaavg', ref_cols=(0, 4, 1))


@pytest.mark.parametrize('isa,nthreads', generate_isa_and_nthreads_combos())
def test_vpf_mocks(Mr19_mock_northonly, isa, nthreads):
    from Corrfunc.mocks import vpf_mocks

    # Max. sphere radius of 10 Mpc
    rmax = 10.0
    # 10 bins..so counts in spheres of radius 1, 2, 3, 4...10 Mpc spheres
    nbin = 10
    num_spheres = 10000
    num_pN = 6
    threshold_neighbors = 1  # does not matter since we have the centers
    centers_file = pjoin(dirname(abspath(__file__)),
                         "../../mocks/tests/data/",
                         "Mr19_centers_xyz_forVPF_rmax_10Mpc.txt")

    binfile = pjoin(dirname(abspath(__file__)),
                    "../../mocks/tests/", "angular_bins")

    x, y, z, w = Mr19_mock_northonly
    results_vpf_mocks = vpf_mocks(rmax, nbin, num_spheres, num_pN,
                                  threshold_neighbors, centers_file,
                                  x, y, z, x, y, z,
                                  verbose=True, isa=isa,)

    file_ref = pjoin(dirname(abspath(__file__)),
                     "../../mocks/tests/", "Mr19_mock_vpf")
    check_vpf_against_reference(results_vpf_mocks, file_ref)
