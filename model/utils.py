#!/usr/bin/env python

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy import stats as scistats
from scipy import interpolate
from scipy.interpolate import interp1d, LinearNDInterpolator, \
    NearestNDInterpolator, CloughTocher2DInterpolator
import scipy.optimize
import astropy.cosmology
import gwdet

cosmo = astropy.cosmology.Planck15  # Use the Planck15 cosmology as implemented in astropy


class hist(object):
    """
    Class to define a multi-dimensional distribution
    given some histogram data.
    This extends the scipy-stats function 'rv_histogram'
    """

    def __init__(self, edges, heights):

        self.edges = edges # list of arrays for bin edges along each dim
        self.heights = heights # n histogram values -> array of bin heights

        self.nheights = np.abs(self.heights)
        self.nheights = self.nheights - self.nheights.min()
        if len(self.edges) == 1:
            self.dist = scistats.rv_histogram((self.nheights,self.edges[0]))

        elif len(self.edges) == 2:
            #bin_coords = [np.unique(self.edges[:,0]),
            #              np.unique(self.edges[:,1])]
            #bin_widths = [bin_coords[0][1] - bin_coords[0][0],
            #              bin_coords[1][1] - bin_coords[1][0]] # regular grid
            # what about using np.histogramdd ???
            #self.dist = self.nheights / float(np.sum(self.nheights * np.prod(bin_widths)))
            #self._hpdf = np.hstack([0.0, self._hpdf, 0.0])

            # bin-center coordinates
            self.edgesm = []
            for edge in self.edges:
                self.edgesm.append((edge[1:] + edge[:-1]) / 2)
            xgrid, ygrid = np.meshgrid(self.edgesm[0], self.edgesm[1])
            self.bin_coords = np.column_stack([xgrid.ravel(), ygrid.ravel()])

            # remove irregularities
            self.dist = self.nheights.copy()
            self.dist = np.abs(self.dist)
            self.dist -= np.min(self.dist)
            # normalize PDF
            delta_params = np.outer(np.diff(self.edges[0]), np.diff(self.edges[1])).flatten()
            norm = np.sum(self.nheights * delta_params)
            self.dist = self.nheights / norm

    def pdf(self, x):
        """
        PDF of the histogram
        """
        if len(self.edges) == 1:
            return self.dist.pdf(x)

        elif len(self.edges) == 2:
            # may need to modify to implement pdf boundary support
            return self.dist[np.linalg.norm(self.bin_coords[:,:,None] - x.T,axis=1).argmin(axis=0)]

def ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.
    Extracted from scipy.interpolate
    """

    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = np.broadcast_arrays(*points)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        points = np.empty(p[0].shape + (len(points),), dtype=float)
        for j, item in enumerate(p):
            points[...,j] = item
    else:
        points = np.asanyarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points

def ndinterp(points, values, method='linear', fill_value=0.0):
    """
    Interpolate unstructured D-dimensional data. Edited from scistats version.

    Parameters
    ----------
    points : ndarray of floats, shape (n, D)
        Data point coordinates in form of (n, D) array
    values : ndarray of float or complex, shape (n,)
        Data values.
    xi : 2-D ndarray of float or tuple of 1-D array, shape (M, D)
        Points at which to interpolate data.
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of
        ``nearest``
          return the value at the data point closest to
          the point of interpolation.  See `NearestNDInterpolator` for
          more details.
        ``linear``
          tessellate the input point set to n-dimensional
          simplices, and interpolate linearly on each simplex.  See
          `LinearNDInterpolator` for more details.
        ``cubic`` (1-D)
          return the value determined from a cubic
          spline.
        ``cubic`` (2-D)
          return the value determined from a
          piecewise cubic, continuously differentiable (C1), and
          approximately curvature-minimizing polynomial surface. See
          `CloughTocher2DInterpolator` for more details.
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points.  If not provided, then the
        default is ``0.0``. This option has no effect for the
        'nearest' method.

    Returns
    -------
    function
        Interpolant function.
    """

    points = ndim_coords_from_arrays(points)

    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        points = points.ravel()
        # Sort points/values together, necessary as input for interp1d
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        if method == 'nearest':
            fill_value = 'extrapolate'
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
                      fill_value=fill_value)
        return ip
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values)
        return ip
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value)
        return ip
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value)
        return ip
    else:
        raise ValueError("Unknown interpolation method %r for "
                         "%d dimensional data" % (method, ndim))

def generate_redshift(model='comoving', z_min=0.0, z_max=1.0):
    """Generate redshift values for binaries."""

    # Uniform in comoving volume
    if model == 'comoving':

        # Min/max comoving distance
        cd_min = 0.0
        cd_max = cosmo.comoving_distance(z_max).value # Mpc

        while True: # Generate numbers uniforms in comoving distance
            cd = np.linalg.norm(np.random.uniform(0, cd_max, 3)) # x,y,z coordinates on a cube
            if cd < cd_max: # select those within a sphere
                break

        # Convert comoving distance to redshift
        z_sol = scipy.optimize.brentq(lambda z: cosmo.comoving_distance(z).value - \
                                      cd , z_min, z_max)

    else:

        raise ValueError, 'select valid model'

    return z_sol

def compute_chirpmass(m1, m2):
    """Compute binary chirp mass from component masses."""

    return (m1 * m2)**(3./5.) / (m1 + m2)**(1./5.)

def det_weights(m1, m2, N=1):
    """Compute detection weights for distributions of CBC systems."""

    pdet = gwdet.detectability() # Instantiate detection weight class

    m1_many = np.array([m1 for x in range(N)]).flatten() # bootstrap binaries
    m2_many = np.array([m2 for x in range(N)]).flatten()

    z_many = np.array([generate_redshift() for x in m1_many]) # compute redshifts
    det_many = pdet(m1_many, m2_many, z_many) # output weights

    return (compute_chirpmass(m1_many, m2_many), z_many, det_many)
