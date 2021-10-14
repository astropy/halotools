# cython: language_level=2
"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals
cimport numpy as cnp
import numpy as np

__author__ = ('Alan Pearl', )

# an abstract condition class
cdef class Condition:
    cdef bint func(self, cnp.int64_t i, cnp.int64_t j):
        raise NotImplementedError()

# default function: Always add 1 per galaxy in cylinder
cdef class AlwaysTrue(Condition):
    cdef bint func(self, cnp.int64_t i, cnp.int64_t j):
        return True

# Add 1 if mass ratio is within specified bounds; else add 0
cdef class MassFrac(Condition):
    def __init__(self, mass1, mass2, mass_frac_lim,
                 lower_equality=False, upper_equality=False):
        self.mass1 = mass1
        self.mass2 = mass2
        self.fmin = mass_frac_lim[0]
        self.fmax = mass_frac_lim[1]
        self.lower_equality = lower_equality
        self.upper_equality = upper_equality

    cdef bint func(self, cnp.int64_t i, cnp.int64_t j):
        cdef bint test1
        cdef bint test2
        cdef cnp.float64_t mass_frac
        if self.mass1[i] == 0.:
            if self.mass2[j] == 0.:
                mass_frac = 1.
            else:
                mass_frac = np.inf
        else:
            mass_frac = self.mass2[j] / self.mass1[i]

        if self.lower_equality:
            test1 = (self.fmin <= mass_frac)
        else:
            test1 = (self.fmin < mass_frac)
        if self.upper_equality:
            test2 = (mass_frac <= self.fmax)
        else:
            test2 = (mass_frac < self.fmax)

        return (test1 and test2)

def choose_condition(condition, args):
    if condition is None or condition == "always_true":
        return AlwaysTrue()
    elif condition == "mass_frac":
        m1 = np.asarray(args[0], dtype=np.float64)
        m2 = np.asarray(args[1], dtype=np.float64)
        lims = [float(a) for a in args[2]]
        equality = args[3:]
        return MassFrac(m1, m2, lims, *equality)
    else:
        raise ValueError("'condition' must be None or "
                         "a permitted descriptive string")
