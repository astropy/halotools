__all__ = ['projected_correlation']
from itertools import izip
import math
import numpy as np
from scipy.integrate import quad
from fast3tree import fast3tree

def _yield_periodic_points(center, dcorner1, dcorner2, box_size):
    cc = np.array(center)
    flag = (cc+dcorner1 < 0).astype(int) - (cc+dcorner2 >= box_size).astype(int)
    cp = cc + flag*box_size
    a = range(len(cc))
    for j in xrange(1 << len(cc)):
        for i in a:
            if j >> i & 1 == 0:
                cc[i] = center[i]
            elif flag[i]:
                cc[i] = cp[i]
            else:
                break
        else:
            yield cc

def _jackknife_2d_random(rbins, box_size, jackknife_nside):
    def corner_area(x, y):
        a = math.sqrt(1.0-x*x)-y
        b = math.sqrt(1.0-y*y)-x
        theta = math.asin(math.sqrt(a*a+b*b)*0.5)*2.0
        return (a*b + theta - math.sin(theta))*0.5

    def segment_without_corner_area(x, r):
        half_chord = math.sqrt(1.0-x*x)
        return math.acos(x) - x*half_chord \
                - quad(corner_area, 0, min(half_chord, 1.0/r), (x,))[0]*r

    def overlapping_circular_areas(r):
        if r*r >= 2: return 1.0
        return (math.pi - quad(segment_without_corner_area, 0, min(1, 1.0/r), \
                (r,))[0]*4.0*r)*r*r

    overlapping_circular_areas_vec = np.vectorize(overlapping_circular_areas, \
            [float])

    side_length = box_size/float(jackknife_nside)
    square_area = 1.0/float(jackknife_nside*jackknife_nside)
    rbins_norm = rbins/side_length
    annulus_areas = np.ediff1d(overlapping_circular_areas_vec(rbins_norm))
    annulus_areas /= np.ediff1d(rbins_norm*rbins_norm)*math.pi
    return 1.0 - square_area * (2.0 - annulus_areas)

def projected_correlation(points, rbins, zmax, box_size, jackknife_nside=0):
    """
    Calculate the projected correlation function wp(rp) and its covariance 
    matrix for a periodic box, with the plane-parallel approximation and 
    the Jackknife method.

    Parameters
    ----------
    points : array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
        The last column will be used as the redshift distance.
    rbins : array_like
        A 1-d array that has the edges of the rp bins. Must be sorted.
    zmax : float
        The integral of \pi goes from -zmax to zmax (redshift distance).
    box_size : float
        The side length of the periodic box.
    jackknife_nside : int, optional (Default: 0)
        If <= 1 , it will not do Jackknife.

    Returns
    -------
    wp : ndarray
        A 1-d array that has wp. The length of this retured array would be
        len(rbins) - 1.
    wp_cov : ndarray (returned if jackknife_nside > 1)
        The len(wp) by len(wp) covariance matrix of wp.
    """
    points = np.asarray(points)
    s = points.shape
    if len(s) != 2 or s[1] != 3:
        raise ValueError('`points` must be a 2-d array with last dim=3')
    N = s[0]

    rbins = np.asarray(rbins)
    rbins_sq = rbins*rbins
    dcorner2 = np.array([rbins[-1], rbins[-1], zmax])
    dcorner1 = -dcorner2
    if np.any(dcorner2*2 > box_size):
        print "[Warning] box too small!"

    pairs_rand = float(N*N) / box_size**3 \
            * (rbins[1:]**2-rbins[:-1]**2)*np.pi*zmax*2.0
    jackknife_nside = int(jackknife_nside)

    if jackknife_nside <= 1: #no jackknife
        dcorner1[2] = 0 #save some time
        pairs = np.zeros(len(rbins)-1, dtype=int)
        with fast3tree(points) as tree:
            for p in points:
                for pp in _yield_periodic_points(p,dcorner1,dcorner2,box_size):
                    x,y=tree.query_box(pp+dcorner1,pp+dcorner2,output='p').T[:2]
                    x -= pp[0]; x *= x
                    y -= pp[1]; y *= y
                    x += y; x.sort()
                    pairs += np.ediff1d(np.searchsorted(x, rbins_sq))
        return (pairs.astype(float)*2.0/pairs_rand - 1.0) * zmax*2.0

    else: #do jackknife
        jack_ids  = np.floor(np.remainder(points[:,0], box_size)\
                / box_size*jackknife_nside).astype(int)
        jack_ids += np.floor(np.remainder(points[:,1], box_size)\
                / box_size*jackknife_nside).astype(int) * jackknife_nside
        n_jack = jackknife_nside*jackknife_nside
        pairs = np.zeros((n_jack, len(rbins)-1), dtype=int)
        auto_pairs = np.zeros_like(pairs)
        with fast3tree(points) as tree:
            for p, jid in izip(points, jack_ids):
                for pp in _yield_periodic_points(p,dcorner1,dcorner2,box_size):
                    idx,pos = tree.query_box(pp+dcorner1,pp+dcorner2,output='b')
                    x, y = pos.T[:2]
                    x -= pp[0]; x *= x
                    y -= pp[1]; y *= y
                    x += y 
                    y = x[jack_ids[idx]==jid]
                    y.sort(); x.sort()
                    pairs[jid] += np.ediff1d(np.searchsorted(x, rbins_sq))
                    auto_pairs[jid] += np.ediff1d(np.searchsorted(y, rbins_sq))
        idx = pos = x = y = None
        pairs_sum = pairs.sum(axis=0)
        pairs = pairs_sum - pairs*2 + auto_pairs
        wp_jack = (pairs.astype(float) \
                / pairs_rand \
                / _jackknife_2d_random(rbins, box_size, jackknife_nside)\
                - 1.0) * zmax*2.0
        wp_full = (pairs_sum.astype(float)/pairs_rand - 1.0) * zmax*2.0
        wp = wp_full*n_jack - wp_jack.mean(axis=0)*(n_jack-1)
        wp_cov = np.cov(wp_jack, rowvar=0, bias=1)*(n_jack-1)
        return wp, wp_cov


def correlation3d(points, rbins, box_size):
    """
    Calculate the 3D correlation function xi(r) for a periodic box.

    Parameters
    ----------
    points : array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns).
    rbins : array_like
        A 1-d array that has the edges of the rp bins. Must be sorted.
    box_size : float
        The side length of the periodic box.

    Returns
    -------
    xi : ndarray
        A 1-d array that has wp. The length of this retured array would be
        len(rbins) - 1.
    """
 
    points = np.asarray(points)
    s = points.shape
    if len(s) != 2 or s[1] != 3:
        raise ValueError('`points` must be a 2-d array with last dim=3')
    N = s[0]

    rbins = np.asarray(rbins)
    pairs_rand = float(N*N) / box_size**3 \
            * (rbins[1:]**3-rbins[:-1]**3)*(np.pi*4.0/3.0)
    
    pairs = np.zeros(len(rbins)-1, dtype=int)
    with fast3tree(points) as tree:
        tree.set_boundaries(0, box_size)
        for p in points:
            pairs += np.ediff1d([tree.query_radius(p, r, periodic=True, \
                    output='c') for r in rbins])

    return pairs.astype(float)/pairs_rand - 1.0
