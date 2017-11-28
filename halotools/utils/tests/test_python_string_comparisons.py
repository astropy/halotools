"""
"""
from ..python_string_comparisons import _passively_decode_string

__all__ = ('test_passively_decode_string', )


def test_passively_decode_string():
    __ = _passively_decode_string('a')
    __ = _passively_decode_string(1)
