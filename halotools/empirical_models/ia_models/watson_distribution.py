r"""
Dimroth-Watson distribution class
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from scipy.stats import rv_continuous
from scipy.special import erf, erfi

from warnings import warn


__all__ = ('DimrothWatson')
__author__ = ('Duncan Campbell')


class DimrothWatson(rv_continuous):
    r"""
    A Dimroth-Watson distribution of :math:`\cos(\theta)'

    Parameters
    ----------
    k : float
        shape paramater

    Notes
    -----
    The Dimroth-Watson distribution is defined as:

    .. math::
        p(\cos(\theta)) = B(k)\exp[-k\cos(\theta)^2]\mathrm{d}\cos(\theta)

    where

    .. math::
        B(k) = \frac{1}{2}int_0^1\exp(-k t^2)\mathrm{d}t

    We assume the ISO convention for spherical coordinates, where :math:`\theta`
    is the polar angle, bounded between :math:`[-\pi, \pi]`, and :math:`\phi`
    is the azimuthal angle, where for a Dimroth-Watson distribution, :math:`phi'
    is a uniform random variable between :math:`[0, 2\pi]`: for all `k`.

    For :math:`k<0`, the distribution of points on a sphere is bipolar.
    For :math:`k=0`, the distribution of points on a sphere is uniform.
    For :math:`k>0`, the distribution of points on a sphere is girdle.

    Note that as :math:`k \rarrow \infty`:

    .. math::
        p(\cos(\theta)) = \frac{1}{2}\left[ \delta(\cos(\theta) + 1) + \delta(\cos(\theta) - 1) \right]\mathrm{d}\cos(\theta)

    and as :math:`k \rarrow -\infty`:

    .. math::
        p(\cos(\theta)) = \frac{1}{2}\delta(\cos(\theta))\mathrm{d}\cos(\theta)

    Needless to say, for large :math:`|k|`, the attributes of this class are approximate and not well tested.
    """

    def _argcheck(self, k):
        r"""
        check arguments
        """
        k = np.asarray(k)
        self.a = -1.0  # lower bound
        self.b = 1.0  # upper bound
        return (k == k)

    def _norm(self, k):
        r"""
        normalization constant
        """

        k = np.atleast_1d(k)

        # mask for positive and negative k cases
        negative_k = (k < 0) & (k != 0)
        positive_k = (k != 0)

        # after masking, ignore the sign of k
        k = np.fabs(k)

        # create an array to store the result
        norm = np.zeros(len(k))

        # for k>0
        norm[positive_k] = 4.0*np.sqrt(np.pi)*erf(np.sqrt(k[positive_k]))/(4.0*np.sqrt(k[positive_k]))
        # for k<0
        norm[negative_k] = 4.0*np.sqrt(np.pi)*erfi(np.sqrt(k[negative_k]))/(4.0*np.sqrt(k[negative_k]))

        # ignore divide by zero in the where statement
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(k == 0, 0.5, 1.0/norm)

    def _pdf(self, x, k):
        r"""
        probability distribution function

        Parameters
        ----------
        k : float
            shape parameter

        Notes
        -----
        See the 'notes' section of the class for a discussion of large :math:`|k|`.
        """

        # process arguments
        k = np.atleast_1d(k).astype(np.float64)
        x = np.atleast_1d(x).astype(np.float64)

        with np.errstate(over='ignore', invalid='ignore'):
            norm = self._norm(k)
            p = norm*np.exp(-1.0*k*x**2)
            p = np.nan_to_num(p)

        # deal with the edge cases
        epsilon = 1e-5
        edge_mask = (p >= 1.0/epsilon) | (p == 0.0)
        #edge_mask = (p >= 1.0/epsilon)
        p[edge_mask] = 0.0

        # large negative k (bipolar)
        bipolar = (x >= (1.0 - epsilon)) | (x <= (-1.0 + epsilon))
        #p[bipolar & edge_mask & (k>1)] = 1.0/(2.0*epsilon)
        p[bipolar & edge_mask & (k < -1)] = 1.0/(2.0*epsilon)

        # large positive k (girdle)
        girdle = (x >= (0.0 - epsilon)) & (x <= (0.0 + epsilon))
        #p[girdle & edge_mask & (k <- 1)] = 1.0/(2.0*epsilon)
        p[girdle & edge_mask & (k > 1)] = 1.0/(2.0*epsilon)

        return p

    def _rvs(self, k, size=None, max_iter=100, random_state=None):
        r"""
        random variate sampling

        Parameters
        ----------
        k : array_like
            array of shape parameters

        size : int or tuple of ints, optional
            integer indicating the number of samples to draw.
            if not given, the number of samples will be equal to len(k).

        max_iter : int, optional
            integer indicating the maximum number of times to iteratively draw from
            the proposal distribution until len(s) points are accepted.
        
        random_state : numpy.random.RandomState, optional
            RandomState used to generate random numbers.

        Notes
        -----
        The random variate sampling for this distribution is an implementation
        of the rejection-sampling technique.

        The Proposal distributions are taken from Best & Fisher (1986).
        """

        k = np.atleast_1d(k).astype(np.float64)
        if size is None or size == ():
            size = len(k)
        if size != 1:
            # If size is an int, the first condition must be met, if size is a tuple, the second condition is the equivalent form
            if len(k) == size or k.shape == size:
                pass
            elif len(k) == 1:
                k = np.ones(size)*k
            else:
                msg = ('if `size` argument is given, len(k) must be 1 or equal to size.')
                raise ValueError(msg)
        else:
            size = len(k)

        # vector to store random variates
        result = np.zeros(size)

        # take care of k=0 case
        zero_k = (k == 0)
        uran0 = random_state.random(np.sum(zero_k))*2 - 1.0
        result[zero_k] = uran0

        # take care of edge cases, i.e. |k| very large
        with np.errstate(over='ignore'):
            x = np.exp(k)
        edge_mask = ((x == np.inf) | (x == 0.0))
        #result[edge_mask & (k>0)] = np.random.choice([1,-1], size=np.sum(edge_mask & (k>0)))
        #result[edge_mask & (k<0)] = 0.0
        result[edge_mask & (k<0)] = random_state.choice([1,-1], size=np.sum(edge_mask & (k<0)))
        result[edge_mask & (k>0)] = 0.0

        # apply rejection sampling technique to sample from pdf
        n_sucess = np.sum(zero_k) + np.sum(edge_mask)  # number of sucesessful draws from pdf
        n_remaining = size - n_sucess  # remaining draws necessary
        n_iter = 0  # number of sample-reject iterations
        kk = k[(~zero_k) & (~edge_mask)]  # store subset of k values that still need to be sampled
        mask = np.repeat(False,size)  # mask indicating which k values have a sucessful sample
        mask[zero_k] = True

        while (n_sucess < size) & (n_iter < max_iter):
            # get three uniform random numbers
            uran1 = random_state.random(n_remaining)
            uran2 = random_state.random(n_remaining)
            uran3 = random_state.random(n_remaining)

            # masks indicating which envelope function is used
            negative_k = (kk < 0.0)
            positive_k = (kk > 0.0)

            # sample from g(x) to get y
            y = np.zeros(n_remaining)
            y[positive_k] = self.g1_isf(uran1[positive_k], kk[positive_k])
            y[negative_k] = self.g2_isf(uran1[negative_k], kk[negative_k])
            y[uran3 < 0.5] = -1.0*y[uran3 < 0.5]  # account for one-sided isf function

            # calculate M*g(y)
            g_y = np.zeros(n_remaining)
            m = np.zeros(n_remaining)
            g_y[positive_k] = self.g1_pdf(y[positive_k], kk[positive_k])
            g_y[negative_k] = self.g2_pdf(y[negative_k], kk[negative_k])
            m[positive_k] = self.m1(kk[positive_k])
            m[negative_k] = self.m2(kk[negative_k])

            # calulate f(y)
            f_y = self.pdf(y, kk)

            # accept or reject y
            keep = ((f_y/(g_y*m)) > uran2)

            # count the number of succesful samples
            n_sucess += np.sum(keep)

            # store y values
            result[~mask] = y

            # update mask indicating which values need to be redrawn
            mask[~mask] = keep

            # get subset of k values which need to be sampled.
            kk = kk[~keep]

            n_iter += 1
            n_remaining = np.sum(~keep)

        if (n_iter == max_iter):
            msg = ('The maximum number of iterations reached, random variates may not be represnetitive.')
            raise warn(msg)

        return result

    def g1_pdf(self, x, k):
        r"""
        proposal distribution for pdf for k>0
        """
        k = -1*k
        eta = np.sqrt(-1*k)
        C = eta/(np.arctan(eta))
        return (C/(1+eta**2*x**2))/2.0

    def g1_isf(self, y, k):
        r"""
        inverse survival function of proposal distribution for pdf for k>0
        """
        k = -1*k
        eta = np.sqrt(-1*k)
        return (1.0/eta)*(np.tan(y*np.arctan(eta)))

    def m1(self, k):
        r"""
        eneveloping factor for proposal distribution for k>0
        """
        return 2.0*np.ones(len(k))

    def g2_pdf(self, x, k):
        r"""
        proposal distribution for pdf for k<0
        """
        k = -1*k
        norm = 2.0*(np.exp(k)-1)/k
        return (np.exp(k*np.fabs(x)))/norm

    def g2_isf(self, y, k):
        r"""
        inverse survival function of proposal distribution for pdf for k<0
        """
        k = -1.0*k
        C = k/(np.exp(k)-1.0)
        return np.log(k*y/C+1)/k

    def m2(self, k):
        r"""
        eneveloping factor for proposal distribution for pdf for k<0
        """
        k = -1.0*k
        C = k*(np.exp(k)-1)**(-1)
        norm = 2.0*(np.exp(k)-1)/k
        return C*norm
