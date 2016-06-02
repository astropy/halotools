"""
This module contains general purpose helper functions
used to provide convenience wrappers for mock objects
"""

import numpy as np

from ...custom_exceptions import HalotoolsError

__all__ = ('three_dim_pos_bundle', 'infer_mask_from_kwargs')


def three_dim_pos_bundle(table, key1, key2, key3,
        return_complement=False, **kwargs):
    """
    Method returns 3d positions of particles in
    the standard form of the inputs used by many of the
    functions in the `~halotools.mock_observables`.

    Parameters
    ----------
    table : data table
        `~astropy.table.halo_table` object

    key1, key2, key3: strings
        Keys used to access the relevant columns of the data table.

    mask : array, optional
        array used to apply a mask over the input ``table``. Default is None.

    return_complement : bool, optional
        If set to True, method will also return the table subset given by the inverse mask.
        Default is False.

    """
    if 'mask' in list(kwargs.keys()):
        mask = kwargs['mask']
        x, y, z = table[key1][mask], table[key2][mask], table[key3][mask]
        if return_complement is True:
            x2, y2, z2 = table[key1][np.invert(mask)], table[key2][np.invert(mask)], table[key3][np.invert(mask)]
            return np.vstack((x, y, z)).T, np.vstack((x2, y2, z2)).T
        else:
            return np.vstack((x, y, z)).T
    else:
        x, y, z = table[key1], table[key2], table[key3]
        return np.vstack((x, y, z)).T


def infer_mask_from_kwargs(galaxy_table, **kwargs):
    """
    """
    if 'mask_function' in kwargs:
        func = kwargs['mask_function']
        mask = func(galaxy_table)
    else:
        galaxy_table_keyset = set(galaxy_table.keys())
        kwargs_set = set(kwargs.keys())
        masking_keys = list(galaxy_table_keyset.intersection(kwargs_set))
        if len(masking_keys) == 0:
            mask = np.ones(len(galaxy_table), dtype=bool)
        elif len(masking_keys) == 1:
            key = masking_keys[0]
            mask = galaxy_table[key] == kwargs[key]
        else:
            # We were passed too many keywords - raise an exception
            msg = ("Only a single mask at a time is permitted by calls to "
                "compute_galaxy_clustering. \nChoose only one of the following keyword arguments:\n")
            arglist = ''
            for arg in masking_keys:
                arglist = arglist + arg + ', '
            arglist = arglist[:-2]
            msg = msg + arglist
            raise HalotoolsError(msg)
    return mask
