"""
This script generates positions, weights, and bins.

```
python prepare_tests.py
```
"""

import numpy as np


def save_bins(fn='bins.txt'):
    size = 20
    bins = np.linspace(0., 100, size + 1)
    bins = list(zip(bins[:-1], bins[1:]))
    np.savetxt(fn, bins)


def save_catalogs(pfn='xyz.txt', wfn='w.txt', size=int(1e6), boxsize=(2000.,) * 3, boxcenter=(5000.,) * 3, n_individual_weights=1, n_bitwise_weights=4, seed=42):
    rng = np.random.RandomState(seed=seed)
    positions = [c + rng.uniform(-0.5, 0.5, size) * s for c, s in zip(boxcenter, boxsize)]
    # integer weights, viewed as doubles
    weights = [rng.randint(0, 0xffffffff, size, dtype='i8').view(dtype='f8') for i in range(n_bitwise_weights)]
    weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
    np.savetxt(pfn, np.column_stack(positions))
    np.savetxt(wfn, np.column_stack(weights))


def save_pair_weights(fn='pair_weights.txt'):
    # cosine angle of pair separation, weight value
    size = 10
    weights = [np.linspace(0.9999, 1., size), np.linspace(3., 2., size)]
    np.savetxt(fn, np.column_stack(weights))


if __name__ == '__main__':

    save_bins()
    save_catalogs(wfn='wbit.txt', n_bitwise_weights=4)
    save_catalogs(wfn='wind.txt', n_bitwise_weights=0)
    save_pair_weights()