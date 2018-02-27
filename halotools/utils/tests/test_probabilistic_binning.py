"""
"""
import numpy as np

from ..probabilistic_binning import fuzzy_digitize

__all__ = ('test1', )


def test1():
    """ Enforce assigned centroid span the sensible range
    """
    npts = int(1e5)
    x = np.random.uniform(-1, 1, npts)
    nbins = 100
    centroids = np.linspace(-2, 2, nbins)

    centroid_numbers = fuzzy_digitize(x, centroids)
    assert centroid_numbers.shape == (npts, )
    assert np.all(centroid_numbers >= 0)
    assert np.all(centroid_numbers <= nbins-1)


def test2():
    """ Enforce no assigned centroid has fewer than min_counts elements
    """
    npts = 100
    x = np.random.uniform(-1, 1, npts)
    nbins = 100
    centroids = np.linspace(-2, 2, nbins)

    centroid_numbers = fuzzy_digitize(x, centroids, min_counts=2)
    uvals, counts = np.unique(centroid_numbers, return_counts=True)
    assert np.all(counts >= 2)

    centroid_numbers = fuzzy_digitize(x, centroids, min_counts=0)
    uvals, counts = np.unique(centroid_numbers, return_counts=True)
    assert np.any(counts == 1)


def test3():
    """ Enforce deterministic behavior when passing a seed
    """
    npts = 100
    x = np.random.uniform(-1, 1, npts)
    nbins = 100
    centroids = np.linspace(-2, 2, nbins)

    centroid_numbers_1 = fuzzy_digitize(x, centroids, min_counts=2, seed=8)
    centroid_numbers_2 = fuzzy_digitize(x, centroids, min_counts=2, seed=8)
    centroid_numbers_3 = fuzzy_digitize(x, centroids, min_counts=2, seed=9)
    assert np.all(centroid_numbers_1==centroid_numbers_2)
    assert not np.all(centroid_numbers_1==centroid_numbers_3)


def test4():
    """ Enforce centroid assignment preferentially assigns membership to bins that are nearby.
    """
    npts = int(1e6)
    x = np.random.uniform(-1, 1, npts)
    nbins = 10
    centroids = np.linspace(-1.01, 1.01, nbins)

    centroid_numbers = fuzzy_digitize(x, centroids, min_counts=2, seed=43)

    itest = 4
    test_mask = (x >= centroids[itest]) & (x < centroids[itest+1])
    assert set(centroid_numbers[test_mask]) == set((itest, itest+1))
    assert np.allclose(np.mean(centroid_numbers[test_mask]), itest+0.5, rtol=0.1)

    dx_bin = centroids[itest+1] - centroids[itest]
    test_mask2 = test_mask & (x < centroids[itest] + dx_bin/10.)
    assert np.mean(centroid_numbers[test_mask2]) < itest + 0.25
    test_mask3 = test_mask & (x > centroids[itest+1] - dx_bin/10.)
    assert np.mean(centroid_numbers[test_mask3]) > itest + 1 - 0.25


def test5():
    """ Enforce centroid assignment is deterministic for points that are coincident with a centroid
    """
    npts = int(1e6)
    x = np.random.uniform(-1, 1, npts)
    nbins = 10
    centroids = np.linspace(-1.01, 1.01, nbins)
    random_mask = np.random.randint(0, 2, npts).astype(bool)
    x[random_mask] = np.random.choice(centroids[1:-1], np.count_nonzero(random_mask))

    centroid_numbers = fuzzy_digitize(x, centroids, min_counts=2, seed=43)

    for i, center in enumerate(centroids):
        mask = x == center
        assert np.all(centroid_numbers[mask] == i)
