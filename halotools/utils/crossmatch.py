""" Module containing the `~halotools.utils.crossmatch` function used to 
calculate indices providing the correspondence between two data tables 
sharing a common objectID.
""" 
import numpy as np 

def crossmatch(x, y, skip_bounds_checking = False):
    """
    Function used to identify elements of an integer array x 
    that may appear in an integer array y, and return the 
    indices providing the correspondence. 

    The applications of this function envolve cross-matching 
    two catalogs/data tables which share an objectID. 
    For example, if you have a primary data table and a secondary data table containing 
    supplementary information about (some of) the objects, the 
    `~halotools.utils.crossmatch` function can be used to "value-add" the 
    primary table with data from the second. 

    For another example, suppose you have a single data table 
    with an object ID column and also a column for a "host" ID column 
    (e.g., ``halo_hostid`` in Halotools-provided catalogs), 
    you can use the `~halotools.utils.crossmatch` function to create new columns 
    storing properties of the associated host. 

    See :ref:`crossmatching_halo_catalogs` for a tutorial on common usage 
    of this function with a halo catalog. 

    Parameters 
    ----------
    x : integer array
        Array of integers with possibly repeated entries. 

    y : integer array
        Array of unique integers. 

    skip_bounds_checking : bool, optional 
        If set to False, the input y will be 

    Returns 
    -------
    idx_x : integer array
        Boolean array used to apply a mask to x 
        such that x[idx_x] == y[idx_y] 

    y_idx : integer array  
        Array of integers used to apply a fancy-indexing mask to y 
        such that x[idx_x] == y[idx_y] 

    Examples 
    --------
    Create a sample of 1e6 random integers between 0 and 1000:

    >>> xmax = 1000
    >>> numx = 1e6
    >>> x = np.random.random_integers(0, xmax, numx)
    >>> x.sort()

    Randomly select 50 unique integers between 0 and 1000:

    >>> y = np.random.choice(xmax+1, 50)
    >>> y.sort()

    Now find the integers in x for which there are matches in y

    >>> x_idx, y_idx = crossmatch(x, y)

    The indexing arrays ``x_idx`` and ``y_idx`` are such that 
    the following assertion always holds true:

    >>> assert np.all(x[x_idx] == y[y_idx])

    See also 
    ---------
    :ref:`crossmatching_halo_catalogs`
    
    """
    # Ensure inputs are Numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Require that the inputs meet the assumptions of the algorithm
    if skip_bounds_checking is True:
        pass
    else:
        try:
            assert len(set(y)) == len(y)
            assert np.all(np.array(y, dtype = np.int64) == y)
            assert np.shape(y) == (len(y), )
        except:
            msg = ("Input array y must be a 1d sequence of unique integers")
            raise ValueError(msg)
        try:
            assert np.all(np.array(x, dtype = np.int64) == x)
            assert np.shape(x) == (len(x), )
        except:
            msg = ("Input array x must be a 1d sequence of integers")
            raise ValueError(msg)

    # Internally, we will work with sorted arrays, and then undo the sorting at the end
    idx_x_sorted = np.argsort(x)
    idx_y_sorted = np.argsort(y)
    x_sorted = np.copy(x[idx_x_sorted])
    y_sorted = np.copy(y[idx_y_sorted])

    # x may have repeated entries, so find the unique values as well as their multiplicity
    unique_xvals, counts = np.unique(x_sorted, return_counts = True)

    # Determine which of the unique x values has a match in y
    unique_xval_has_match = np.in1d(unique_xvals, y_sorted, assume_unique = True)

    # Create a boolean array with True for each value in x with a match, otherwise False
    idx_x = np.repeat(unique_xval_has_match, counts)

    # For each unique value of x with a match in y, identify the index of the match
    matching_indices_in_y = np.searchsorted(y_sorted, unique_xvals[unique_xval_has_match])

    # Repeat each matching index according to the multiplicity in x
    idx_y = np.repeat(matching_indices_in_y, counts[unique_xval_has_match])

    # Undo the original sorting and return the result
    return idx_x_sorted[idx_x], idx_y_sorted[idx_y]




