""" Subpackage containing modules of functions that calculate many variations
on galaxy/halo alignments.
"""
from __future__ import absolute_import

from .w_gplus import w_gplus
from .w_gminus import w_gminus
from .w_plusplus import w_plusplus
from .w_minusminus import w_minusminus

__all__ = ('w_gplus', 'w_gminus', 'w_plusplus', 'w_minusminus')
