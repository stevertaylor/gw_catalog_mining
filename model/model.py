#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy import special as ss
from scipy import stats as scistats
from scipy.interpolate import interp1d
import random
import utils

class model(object):
    """
    Class to define a Bayesian model, with priors, a likelihood,
    and a function proportional to the posterior PDF.
    """

    def __init__(self, data, x, interp, sampTrans,
                 dataComp, pmin, pmax, yerr=None,
                 interpType='linear1d', interpScale='linear',
                 interpErrors=False, interpHyperErrors=False,
                 catalogType='median', rate_data=None, rate_interp=None,
                 rate_mean=None, rate_std=None, rate_poisson_marg=True,
                 analytic=None):

        self.data = data # binned data
        self.x = x # x-locations (i.e. histogram bin edges)
        self.yerr = yerr # uncertainties on data
        self.sampTrans = sampTrans # instance of sampling transformation object
        self.dataComp = dataComp # instance of data compression object
        self.interp = interp # list of interpolants
        self.interpType = interpType # e.g. linear1d, gp1d, gp2d
        self.interpScale = interpScale # did you train on 'linear' or 'log' data
        self.catalogType = catalogType # 'median', 'samples'
        self.interpErrors = interpErrors # propagate GP interpolation uncertainty
        self.interpHyperErrors = interpHyperErrors # propagate GP hyperparameter uncertainty
        if self.interpHyperErrors:
            if self.interpType == 'gp1d' or self.interpType == 'gp2d':
                self.gp_kernel_map = None # list of GP kernel MAP values
                                          # in log-10 format
                self.gp_kernel_posterior = None # list of GP kernel hyperparameter
                                                # posteriors in log-10 format

        self.pmax = pmax #np.array([np.log10(10.0), np.log10(10.0)]) # sampling ranges
        self.pmin = pmin #np.array([np.log10(0.01), np.log10(0.01)])

        if rate_interp is not None:
            self.rate_data = rate_data
            self.rate_interp = rate_interp
            self.rate_mean = rate_mean
            self.rate_std = rate_std
            self.poisson_marg = rate_poisson_marg
        else:
            self.rate_data = None
            self.rate_interp = None
            self.rate_mean = None
            self.rate_std = None
            self.poisson_marg = rate_poisson_marg

        if analytic is not None:
            self.func = analytic
        else:
            self.func = None

    def lnprior(self, p):
        """
        Standard uniform prior over finite range.

        :return lnp: Natural logarithm of prior pdf
        """
        lnp = 0.
        if np.all(p <= self.pmax) and np.all(p >= self.pmin):
            lnp = np.sum(np.log(1.0/(self.pmax-self.pmin)))
        else:
            lnp = -np.inf
        return lnp

    def lnlike(self, p):

        lnlike = 0.0
        # need to generalize the following!!
        if self.func is not None:
            lnlike += np.sum(np.log(self.func(self.data[:,0], self.data[:,1], \
                                              10.0**p[0], 10.0**p[1])))
        else:
            try:
                if self.interpType == 'linear1d':
                    # For Linear Interpolation
                    # ------------------------
                    pdf = self.dataComp.rotate2full(np.array([self.interp[jj](self.sampTrans.range2unit(10.0**p))
                                                                for jj in range(len(self.interp))]).flatten())

                elif self.interpType == 'gp1d':
                    # For 1D GP
                    # ----------
                    if not self.interpErrors:
                        pdf = self.dataComp.rotate2full(np.array([self.interp[jj].predict(self.dataComp.pca_weights[jj,:],
                                                                                          self.sampTrans.range2unit(10.0**p))[0][0]
                                                                  for jj in range(len(self.interp))]))
                    elif self.interpErrors:
                        if not self.interpHyperErrors:
                            pdf = self.dataComp.rotate2full(np.array([self.interp[jj].sample_conditional(self.dataComp.pca_weights[jj,:],
                                                                                                         self.sampTrans.range2unit(10.0**p))
                                                                      for jj in range(len(self.interp))]).flatten())
                        elif self.interpHyperErrors:
                            # drawing new kernel hyperparameters from posterior
                            [self.interp[jj].set_parameter_vector(random.choice(
                                self.gp_kernel_posterior[jj] / np.log10(np.e)))
                             for jj in range(len(self.interp))]

                            # sampling conditional, as before
                            pdf = self.dataComp.rotate2full(np.array([self.interp[jj].sample_conditional(self.dataComp.pca_weights[jj,:],
                                                                                                         self.sampTrans.range2unit(10.0**p))
                                                                      for jj in range(len(self.interp))]).flatten())

                elif self.interpType == 'gp2d':
                    # For 2D GP
                    # ----------
                    xrot = np.zeros((self.dataComp.user_dim,self.dataComp.user_dim2))
                    for ii in range(self.dataComp.user_dim):
                        for jj in range(self.dataComp.user_dim2):
                            if not self.interpErrors:
                                xrot[ii,jj] = self.interp[ii][jj].predict(self.dataComp.unitCore[ii,jj,:],
                                                                          self.sampTrans.range2unit([10.0**p]))[0][0]
                            elif self.interpErrors:
                                if not self.interpHyperErrors:
                                    xrot[ii,jj] = self.interp[ii][jj].sample_conditional(self.dataComp.unitCore[ii,jj,:],
                                                                                         self.sampTrans.range2unit([10.0**p]))
                                elif self.interpHyperErrors:
                                    # drawing new kernel hyperparameters from posterior
                                    self.interp[ii][jj].set_parameter_vector(random.choice(
                                        self.gp_kernel_posterior[ii][jj] / np.log10(np.e)))

                                    # sampling conditional, as before
                                    xrot[ii,jj] = self.interp[ii][jj].sample_conditional(self.dataComp.unitCore[ii,jj,:],
                                                                                         self.sampTrans.range2unit([10.0**p]))

                    pdf = self.dataComp.rotate2full(xrot).flatten(order='F')

                # did you train on the distribution ('linear') or
                # 'log10' of the distribution.
                if self.interpScale == 'linear':
                    pdf = pdf
                elif self.interpScale == 'log10':
                    pdf = 10.0**pdf

                # construct normalized PDF
                pdf = utils.hist(self.x, pdf)
                # query PDF at data locations
                if self.catalogType == 'median':
                    pdf_val = pdf.pdf(self.data)
                elif self.catalogType == 'samples':
                    try:
                        pdf_val = np.mean(pdf.pdf(self.data),axis=0)
                    except ValueError:
                        # array indexing: [sample, source, parameter]
                        pdf_val = np.mean([pdf.pdf(self.data[kk])
                                           for kk in range(self.data.shape[0])],
                                          axis=0)

                # incoporate expected rate information
                if self.rate_interp is not None:
                    # rate = self.rate_interp.predict(self.rate_data,
                    #                                 self.sampTrans.range2unit(np.atleast_2d(10.0**p)))[0][0]
                    if not self.interpErrors:
                        rate = self.rate_interp.predict(self.rate_data,
                                                        self.sampTrans.range2unit(np.atleast_2d(10.0**p)))[0][0]
                    elif self.interpErrors:
                        if not self.interpHyperErrors:
                            rate = self.rate_interp.sample_conditional(self.rate_data,
                                                                       self.sampTrans.range2unit(np.atleast_2d(10.0**p)))
                        elif self.interpHyperErrors:
                            # drawing new kernel hyperparameters from posterior
                            self.rate_interp.set_parameter_vector(random.choice(
                                self.rate_gp_kernel_posterior/ np.log10(np.e)))

                            rate = self.rate_interp.sample_conditional(self.rate_data,
                                                                       self.sampTrans.range2unit(np.atleast_2d(10.0**p)))

                    rate = 10.0**(self.rate_mean + self.rate_std * rate)
                    pdf_val *= rate

                lnlike += np.sum(np.log(pdf_val))

                if self.rate_interp is not None:
                    if self.poisson_marg:
                        lnlike -= (1.0 + self.data.shape[0]) * np.log(rate)
                        if rate < 1e-5: lnlike = -np.inf
                    elif not self.poisson_marg:
                        lnlike -= rate

                if np.isnan(lnlike):
                    lnlike = -np.inf
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
