""" Module stores functions used to load the binary outputs of UniverseMachine into memory
"""
import numpy as np


default_sfr_catalog_dtype = np.dtype([('id', '<i8'), ('descid', '<i8'), ('upid', '<i8'),
    ('flags', '<i4'), ('uparent_dist', '<f4'), ('pos', '<f4', (6,)),
    ('vmp', '<f4'), ('lvmp', '<f4'), ('mp', '<f4'), ('m', '<f4'), ('v', '<f4'),
    ('r', '<f4'), ('rank1', '<f4'), ('rank2', '<f4'), ('ra', '<f4'),
    ('rarank', '<f4'), ('t_tdyn', '<f4'), ('sm', '<f4'), ('icl', '<f4'),
    ('sfr', '<f4'), ('obs_sm', '<f4'), ('obs_sfr', '<f4'), ('obs_uv', '<f4'), ('foo', '<f4')])


__all__ = ('load_um_binary_sfr_catalog', )


def load_um_binary_sfr_catalog(fname, dtype=default_sfr_catalog_dtype):
    """ Read the binary UniverseMachine outputs sfr_catalog_XXX.bin into
    a Numpy structured array.

    The returned data structure contains every UniverseMachine galaxy at the
    redshift of the snapshot.

    Parameters
    ----------
    fname : string
        Absolute path to the binary file

    dtype : Numpy dtype, optional
        Numpy dtype defining the format of the returned structured array.

        The default option is compatible with the particular
        outputs used in the first UniverseMachine data release.
        In general, the ``dtype`` argument must be compatible with the
        ``catalog_halo`` struct declared in the make_sf_catalog.h UniverseMachine header.

    Returns
    -------
    arr : Numpy structured array
        UniverseMachine mock galaxy catalog at a single snapshot
    """
    return np.fromfile(fname, dtype=dtype)
