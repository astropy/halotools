"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.integrate import quad


from ..watson_distribution import DimrothWatson
from ..ia_strength_models import alignment_strength

__all__ = ('test_pdf',
           'test_cdf',
           'test_rvs',
           'test_rvs_edge_cases',
           'test_pdf_edge_cases')

def test_pdf():
    """
    test pdf attribute
    """

    d = DimrothWatson()

    # test PDF
    k = 1
    P = quad(d.pdf, -1, 1, args=(k,))[0]
    assert np.isclose(P, 1.0)

    k = 0
    P = quad(d.pdf, -1, 1, args=(k,))[0]
    assert np.isclose(P, 1.0)

    k = -1
    P = quad(d.pdf, -1, 1, args=(k,))[0]
    assert np.isclose(P, 1.0)


def test_cdf():
    """
    test cdf attribute
    """

    d = DimrothWatson()

    # test CDF
    k = 1
    P1 = d.cdf(-1, k=k)
    P2 = d.cdf(1, k=k)
    assert np.isclose(P1, 0.0) & np.isclose(P2, 1.0)

    k = 0
    P1 = d.cdf(-1, k=k)
    P2 = d.cdf(1, k=k)
    assert np.isclose(P1, 0.0) & np.isclose(P2, 1.0)

    k = -1
    P1 = d.cdf(-1, k=k)
    P2 = d.cdf(1, k=k)
    assert np.isclose(P1, 0.0) & np.isclose(P2, 1.0)


def test_rvs():
    """
    test rvs attribute
    """

    d = DimrothWatson()

    # test rvs
    N = 1000

    k = -1.0
    random_variates = d.rvs(k, size=N)
    assert np.isclose(np.mean(random_variates), 0.0, atol=0.1)

    k = 0.0
    random_variates = d.rvs(k, size=N)
    assert np.isclose(np.mean(random_variates), 0.0, atol=0.1)

    k = 1.0
    random_variates = d.rvs(k, size=N)
    assert np.isclose(np.mean(random_variates), 0.0, atol=0.1)


def test_rvs_edge_cases():

    d = DimrothWatson()

    N = 1000

    k = alignment_strength(1)                   # Perfect alignment
    random_variates = d.rvs(k, size=N)

    # Perfect alignment should give cos(theta) = 1, theta = 0 if perfectly aligned
    assert np.all(np.fabs(random_variates)==1.0)

    k = alignment_strength(-1)                   # Perfect anti-alignment
    random_variates = d.rvs(k, size=N)

    # Perfect anti-alignment should give cos(theta) = 0, theta = 90 degrees if perfectly aligned
    assert np.all(np.fabs(random_variates)==0.0)


def test_pdf_edge_cases():

    d = DimrothWatson()
    epsilon = 1e-5

    x = np.array([-1.0, 0.0, 1.0])

    k = alignment_strength(1)
    p = d.pdf(x, k=k)
    assert np.all(p==[1.0/(2*epsilon), 0.0, 1.0/(2*epsilon)])

    k = alignment_strength(-1)
    p = d.pdf(x, k=k)

    assert np.all(p==[0.0, 1.0/(2*epsilon), 0.0])