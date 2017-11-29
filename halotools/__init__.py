"""
Halotools is a specialized python package
for building and testing models of the galaxy-halo connection,
and analyzing catalogs of dark matter halos.
"""
from ._astropy_init import *

from . import custom_exceptions


def test_installation(*args, **kwargs):
    kwargs.setdefault('args', '')
    if kwargs['args']:
        kwargs['args'] += ' '
    kwargs['args'] += '-m installation_test'
    return test(*args, **kwargs)
