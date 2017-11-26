"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals


__all__ = ('compare_strings_py23_safe', )


def _passively_decode_string(a):
    try:
        return a.decode()
    except AttributeError:
        return a


def compare_strings_py23_safe(a, b):
    """
    """
    a = _passively_decode_string(a)
    b = _passively_decode_string(b)
    return a == b
