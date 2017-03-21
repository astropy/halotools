"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = ('direct_from_halo_catalog', )


def direct_from_halo_catalog(**kwargs):
    """
    Function returns the concentrations that are already present
    in an input halo catalog.

    Parameters
    ----------
    table : object
        `~astropy.table.Table` storing halo catalog.

    concentration_key : string
        Name of the table column storing the concentration

    Returns
    -------
    concentration : array_like
        Concentrations of the input halos.
    """
    try:
        table = kwargs['table']
        concentration_key = kwargs['concentration_key']
    except KeyError:
        msg = ("The ``direct_from_halo_catalog`` function accepts two keyword arguments:\n"
            "``table`` and ``concentration_key``")
        raise KeyError(msg)

    try:
        concentration = table[concentration_key]
    except:
        msg = ("The ``{0}`` key does not appear in the input halo catalog.\n"
            "However, you have selected the ``direct_from_halo_catalog`` option \n"
            "to model the concentration-mass relation.\n"
            "The available keys are:\n\n{1}\n")
        raise KeyError(msg.format(concentration_key, list(table.keys())))

    return concentration
