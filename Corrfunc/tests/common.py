import os
from os.path import dirname, abspath, join as pjoin

import numpy as np
import pytest

from Corrfunc.io import read_fastfood_catalog, read_ascii_catalog

NO_DATA_SKIP_MESSAGE = "Data files for data-based tests not found. " \
    "Install from source to enable these tests."


@pytest.fixture(scope='module')
def gals_Mr19():
    filename = pjoin(dirname(abspath(__file__)),
                     "../../theory/tests/data", "gals_Mr19.ff")
    try:
        x, y, z, w = read_fastfood_catalog(filename)
    except OSError:
        pytest.skip(NO_DATA_SKIP_MESSAGE)

    return x, y, z, w


@pytest.fixture(scope='module')
def Mr19_mock_northonly():
    filename = pjoin(dirname(abspath(__file__)),
                     '../../mocks/tests/data', 'Mr19_mock_northonly.xyz.txt')
    return read_ascii_catalog(filename)  # ra, dec, cz, w


@pytest.fixture(scope='module')
def Mr19_mock_northonly_rdz():
    filename = pjoin(dirname(abspath(__file__)),
                     '../../mocks/tests/data', 'Mr19_mock_northonly.rdcz.txt')
    #                 "../../mocks/tests/data", "Mr19_mock_northonly.rdcz.ff")
    try:
        ra, dec, cz, w = read_ascii_catalog(filename)  # ra, dec, cz, w
        #ra, dec, cz, w = read_fastfood_catalog(filename)
    except OSError:
        pytest.skip(NO_DATA_SKIP_MESSAGE)

    return ra, dec, cz, w


@pytest.fixture(scope='module')
def Mr19_randoms_northonly():
    filename = pjoin(dirname(abspath(__file__)),
                     '../../mocks/tests/data', 'Mr19_randoms_northonly.xyz.txt')
    #                 "../../mocks/tests/data", "Mr19_randoms_northonly.rdcz.ff")
    try:
        ra, dec, cz, w = read_ascii_catalog(filename)
        #ra, dec, cz, w = read_fastfood_catalog(filename)
    except OSError:
        pytest.skip(NO_DATA_SKIP_MESSAGE)

    return ra, dec, cz, w


def maxthreads():
    '''Use as many threads as cores that are available to this process'''
    import multiprocessing

    try:
        maxthreads = len(os.sched_getaffinity(0))
    except AttributeError:
        maxthreads = multiprocessing.cpu_count() or 1

    return maxthreads


def generate_isa_and_nthreads_combos(extra_isa=None):
    '''Test all ISA with maxthreads, and the fastest ISA with threads 1 and maxthreads+1'''
    mx = maxthreads()

    # the ISA sweep will use maxthreads
    # and then with the fastest ISA, we will test single-threaded,
    # plus "oversubscribed", where we use more threads than cores
    all_nthreads = [1, mx + 1]

    combos = []
    all_isa = ['fallback', 'sse42', 'avx', 'avx512f']
    if extra_isa:
        all_isa += extra_isa
    combos += [(isa, mx) for isa in all_isa]
    combos += [('fastest', n) for n in all_nthreads]

    return combos


def check_against_reference(results, filename,
                            ref_cols=(0, 4, 1),
                            ravg_name='ravg',
                            cf_name=None,
                            atol=1e-9, rtol=1e-6):
    # results is output of Python function
    # filename has the reference counts
    # ref_cols in order of npairs, weightavg, rpavg, [xi]

    names = ['npairs', 'weightavg', ravg_name]
    dtypes = [np.int64, np.float64, np.float64]
    if cf_name != None:
        # modules like xi return both npairs and xi, so we'll check both
        names += [cf_name]
        dtypes += [np.float64]

    refs = np.genfromtxt(filename, usecols=ref_cols,
                         names=names, dtype=dtypes,
                        )

    assert (results['npairs'] == refs['npairs']).all()
    for name in names[1:]:
        assert np.allclose(results[name], refs[name], atol=atol, rtol=rtol)


def check_vpf_against_reference(results, filename,
                                atol=1e-9, rtol=1e-6):
    # results is output of Python function
    # filename has the reference counts

    numN = results['pN'].shape[-1]
    refs = np.genfromtxt(filename, usecols=range(1, numN + 1), dtype=np.float64)
    assert np.allclose(results['pN'], refs, atol=atol, rtol=rtol)

'''
def convert_ff():

    from cosmoprimo import Cosmology
    # Las Damas cosmology
    cosmo = Cosmology(Omega_m=0.25, Omega_b=0.04, h=0.7, sigma8=0.8, n_s=1., engine='class')
    for filename in ['Mr19_mock_northonly.rdcz.ff', 'Mr19_randoms_northonly.rdcz.ff']:
        filename = pjoin(dirname(abspath(__file__)), '../../mocks/tests/data', filename)
        ra, dec, cz, w = read_fastfood_catalog(filename)
        cz = cosmo.comoving_radial_distance(cz / 299800.0)
        np.savetxt(filename.replace('rdcz.ff', 'rdcz.txt'), np.column_stack([ra, dec, cz, w]))
        conversion = np.pi / 180.
        cos_dec = np.cos(dec * conversion)
        x = cz * cos_dec * np.cos(ra * conversion)
        y = cz * cos_dec * np.sin(ra * conversion)
        z = cz * np.sin(dec * conversion)
        np.savetxt(filename.replace('rdcz.ff', 'xyz.txt'), np.column_stack([x, y, z, w]))


if __name__ == '__main__':

    convert_ff()
'''
