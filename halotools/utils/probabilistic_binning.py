""" Functions used to probabilistically digitize an array
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


__all__ = ('fuzzy_digitize', )


def fuzzy_digitize(x, centroids, min_counts=2, seed=43):
    """ Function assigns each element of the input array ``x`` to a centroid number.

    Centroid-assignment is probabilistic. When a point in ``x`` is halfway between two centroids,
    it is equally likely to be assigned to the centroid to its left or right;
    when a point in ``x`` is coincident with a centroid,
    it will be assigned to that centroid with unit probability; assignment probability
    increases linearly as points approach a centroid.

    The `fuzzy_digitize` function optionally enforces that elements of very sparsely
    populated bins are remapped to the nearest bin with more than ``min_counts`` elements.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ) storing the values to be binned

    centroids : ndarray
        Numpy array of shape (num_centroids, ). The values of ``centroids`` must
        strictly encompass the range of values spanned by ``x`` and must also be
        monotonically increasing.

    min_counts : int, optional
        Minimum required number of elements assigned to each centroid.
        For those centroids not satisfying this requirement,
        all their elements will be reassigned to the nearest sufficiently populated centroid.
        Default is two.

    seed : int, optional
        Random number seed. Default is 43.

    Returns
    -------
    centroid_indices : ndarray
        Numpy integer array of shape (npts, ) storing the index of the centroid
        to which elements of ``x`` are assigned. All integer values of ``centroid_indices``
        will lie in the closed interval [0, num_centroids-1].

    Examples
    --------
    >>> npts = int(1e5)
    >>> xmin, xmax = 0, 8
    >>> x = np.random.uniform(xmin, xmax, npts)
    >>> epsilon, nbins = 0.001, 5
    >>> xbin_edges = np.linspace(xmin-epsilon, xmax+epsilon, nbins)
    >>> centroid_indices = fuzzy_digitize(x, xbin_edges)

    .. image:: /_static/fuzzy_binning_example.png

    """
    assert np.all(np.diff(centroids) > 0), "centroids must be monotonically increasing"
    assert centroids[0] < x.min(), "smallest bin must be less than smallest element in x"
    assert centroids[-1] > x.max(), "largest bin must be less than largest element in x"

    npts_x = len(x)
    num_centroids = len(centroids)
    centroid_indices = np.zeros_like(x).astype(int)-999

    with NumpyRNGContext(seed):
        uran = np.random.rand(npts_x)

    for i, low, high in zip(np.arange(num_centroids).astype(int), centroids[:-1], centroids[1:]):
        bin_mask = (x >= low) & (x < high)

        npts_bin = np.count_nonzero(bin_mask)
        if npts_bin > 0:
            x_in_bin = x[bin_mask]
            dx_bin = high - low
            x_in_bin_rescaled = (x_in_bin - low)/float(dx_bin)

            high_bin_selection = (x_in_bin_rescaled > uran[bin_mask])
            bin_assignment = np.zeros(npts_bin).astype(int) + i
            bin_assignment[high_bin_selection] = i + 1
            centroid_indices[bin_mask] = bin_assignment

    centroid_indices[centroid_indices == -999] = 0

    return enforce_bin_counts(centroid_indices, min_counts)


def enforce_bin_counts(centroid_indices, min_counts):
    """ Function enforces that each entry of `centroid_indices` appears at least `min_counts` times.
    For entries not satisfying this requirement, the nearest index of a sufficiently populated bin
    will be used as a replacement.

    Parameters
    ----------
    centroid_indices : ndarray
        Numpy integer array storing bin numbers

    min_counts : int
        Minimum acceptable number of elements per bin

    Returns
    -------
    output_bin_inidices : ndarray
        Numpy integer array storing bin numbers after enforcing the population requirement.

    Examples
    --------
    >>> centroid_indices = np.random.randint(0, 1000, 1000)
    >>> min_counts = 3
    >>> output_centroid_indices = enforce_bin_counts(centroid_indices, min_counts)
    """
    if min_counts == 0:
        return centroid_indices
    else:
        output_centroid_indices = np.copy(centroid_indices)
        unique_bin_numbers, counts = np.unique(centroid_indices, return_counts=True)
        for i, bin_number, count in zip(np.arange(len(counts)), unique_bin_numbers, counts):
            new_bin_number = _find_nearest_populated_bin_number(
                counts, unique_bin_numbers, i, min_counts)
            if new_bin_number != bin_number:
                output_centroid_indices[centroid_indices==bin_number] = new_bin_number
        return output_centroid_indices


def _find_nearest_populated_bin_number(counts, bin_numbers, bin_index, min_counts):
    """ Helper function used by the `enforce_bin_counts` function.
    """
    bin_numbers = np.atleast_1d(bin_numbers)
    centroid_indices = np.arange(len(bin_numbers))
    counts = np.atleast_1d(counts)
    msg = "Must have at least one bin with greater than {0} elements"
    assert np.any(counts >= min_counts), msg.format(min_counts)

    counts_mask = counts >= min_counts
    available_bin_numbers = bin_numbers[counts_mask]
    available_indices = centroid_indices[counts_mask]

    return available_bin_numbers[np.argmin(np.abs(available_indices - bin_index))]


