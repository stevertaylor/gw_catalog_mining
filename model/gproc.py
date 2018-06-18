#!/usr/bin/env python

from __future__ import division
import numpy as np
import george
import george.kernels as kernels

class gp(object):
    """
    A Gaussian Process class containing the kernel parameter priors and
    a log-likelihood function.
    """

    def __init__(self, x, y, p0, pmin, pmax,
                 kernel = 'sqexp', yerr = None):
        """
        :param x: simulation coordinates [in unit hypercube space]
        :param y: data values
        :param p0: initial values of kernel paramters
        :param pmin: lower sampling boundaries of kernel parameters
        :param pmax: upper sampling boundaries of kernel parameters
        :param kernel: string denoting kernel function type [default = sqexp]
        :param yerr: uncertainties on data values
        :return gaussproc: instance of a George GP object
        """
        self.x = x # simulation coordinates
        self.y = y # data values
        self.yerr = yerr # uncertainties
        self.kernel = kernel # kernel-type: other choices are matern32, matern52
        self.gaussproc = None

        self.p0 = p0 # initial value for kernel parameters
        a, tau = 10.0**self.p0[0], 10.0**self.p0[1:]
        if self.kernel == 'sqexp':
            self.gaussproc = george.GP(a * kernels.ExpSquaredKernel(tau,ndim=len(tau)))
        elif self.kernel == 'matern32':
            self.gaussproc = george.GP(a * kernels.Matern32Kernel(tau),ndim=len(tau))
        elif self.kernel == 'matern52':
            self.gaussproc = george.GP(a * kernels.Matern52Kernel(tau),ndim=len(tau))

        self.gaussproc.compute(self.x , self.yerr)

        self.pmax = pmax  # sampling max
        self.pmin = pmin # sampling min
        self.emcee_flatchain = None
        self.emcee_flatlnprob = None
        self.emcee_kernel_map = None

    def lnprior(self, p):
        """
        Standard uniform prior over finite range.

        :return lnp: Natural logarithm of prior pdf
        """
        lnp = 0.0
        if np.all(p <= self.pmax) and np.all(p >= self.pmin):
            lnp = np.sum(np.log(1.0/(self.pmax-self.pmin)))
        else:
            lnp = -np.inf
        return lnp

    def lnlike(self, p):
        """
        GP likelihood function for probability of data given the kernel parameters

        :return lnlike: likelihood of kernel amplitude and length-scale parameters
        """
        # Update the kernel and compute the lnlikelihood.
        a, tau = 10.0**p[0], 10.0**p[1:]

        lnlike = 0.0
        try:

            if self.kernel == 'sqexp':
                self.gaussproc = george.GP(a * kernels.ExpSquaredKernel(tau,ndim=len(tau)))
            elif self.kernel == 'matern32':
                self.gaussproc = george.GP(a * kernels.Matern32Kernel(tau,ndim=len(tau)))
            elif self.kernel == 'matern52':
                self.gaussproc = george.GP(a * kernels.Matern52Kernel(tau,ndim=len(tau)))

            self.gaussproc.compute(self.x , self.yerr)

            lnlike = self.gaussproc.log_likelihood(self.y, quiet=True)

        except np.linalg.LinAlgError:

            lnlike = -np.inf

        return lnlike

    def lnprob(self, p):
        """
        Sum log-prior and log-likelihood together to get a value
        proportional to the log-posterior.

        :return lnpost: proportional to log-posterior
        """
        # Evaluate prior first
        tmp_prior = self.lnprior(p)
        if tmp_prior == -np.inf:
            lnpost = tmp_prior
        else:
            lnpost = tmp_prior + self.lnlike(p)
        return lnpost
