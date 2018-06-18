#!/usr/bin/env python

from __future__ import division
import numpy as np
try:
    import tensorly as tl
    import tensorly.backend as tlb
    import tensorly.decomposition as tld
except ImportError:
    print "No module name tensorly"

class dataCompress(object):
    """Class for dimensional reduction of histogram data using PCA"""

    def __init__(self, dataMat, histBins, simDesign, tol=1e-5, user_dim=None):
        """
        :param dataMat: Matrix of simulation data [nbins x nsims]
        :param histBins: Coordinates of bins
        :param simDesign: Coordinates of simulations
        :param tol: PCA tolerance for singular values [default = 1e-5]
        :param user_dim: Number of singular values to take [default = None]

        :return rowMean: Mean of each histogram bin
        :return rowStd: Standard deviation of each histogram bin
        """
        self.dataMat = dataMat
        self.histBins = histBins
        self.simDesign = simDesign
        self.tol = tol
        self.user_dim = user_dim
        self.dim = None
        self.unitVar = None
        self.rowMean = np.mean(dataMat,axis=-1)
        self.rowStd = np.std(dataMat,axis=-1)

    def unitTrans(self, unitVar=True):
        """
        Convert each histogram bin to zero-mean, unit variance.

        :return unitMat: zero-mean, unit-variance rows (i.e. bins)
        """
        self.unitMat = self.dataMat.copy() - self.rowMean[:,None]
        self.unitVar = unitVar
        if self.unitVar:
            self.unitMat /= self.rowStd[:,None]

    def basisCompute(self):
        """
        PCA compression of simulation data.

        :return pca_basis: Matrix with columns as orthogonal basis vectors
        :return pca_weights: Coefficients of basis vectors at simulation points
        """
        u,s,v = np.linalg.svd(self.unitMat,full_matrices=False)

        if self.user_dim is not None:
            self.pca_basis = 1.0 / np.sqrt(s.shape[0]) * u[:,:self.user_dim] * s[:self.user_dim]
            self.pca_weights = np.sqrt(s.shape[0]) * v[:self.user_dim,:]
        elif self.user_dim is None:
            mask = s/s.max() > self.tol
            self.dim = np.sum(mask)
            self.pca_basis = 1.0 / np.sqrt(s.shape[0]) * u[:,mask] * s[mask]
            self.pca_weights = np.sqrt(s.shape[0]) * v[mask,:]

    def rotate2full(self, vec, error=False):
        """
        Rotate a prediction back into full-rank space (i.e. bin space)

        :param vec: Vector in low-rank space
        :param error: Is this a vector of uncertainties? [default = False]

        :return full_rank: Full rank prediction
        """
        if self.unitVar:
            full_rank = np.dot(self.pca_basis, vec) * self.rowStd + self.rowMean
        else:
            full_rank = np.dot(self.pca_basis, vec) + self.rowMean

        if error:
            full_rank -= self.rowMean

        return full_rank

    def rotate2low(self, vec):
        """
        Rotate a full-rank (i.e. bin space) vector into reduced space

        :return low_rank: Low rank vector
        """
        if self.unitVar:
            low_rank = np.dot(np.linalg.pinv(self.pca_basis), (vec - self.rowMean) / self.rowStd)
        else:
            low_rank = np.dot(np.linalg.pinv(self.pca_basis), vec - self.rowMean)
        return low_rank

    def match(self):
        """
        Return match of un-compressed data against the original data

        :return match: match vector across all simulations
        """
        data_dot_pca = np.array([np.dot(self.dataMat[:,ii].flatten(),
                                        self.rotate2full(self.pca_weights[:,ii]).flatten())
                                 for ii in range(self.dataMat.shape[-1])])
        data_dot_data = np.array([np.dot(self.dataMat[:,ii].flatten(), self.dataMat[:,ii].flatten())
                                  for ii in range(self.dataMat.shape[-1])])
        pca_dot_pca = np.array([np.dot(self.rotate2full(self.pca_weights[:,ii]).flatten(),
                                       self.rotate2full(self.pca_weights[:,ii]).flatten())
                                for ii in range(self.dataMat.shape[-1])])
        return data_dot_pca / np.sqrt(data_dot_data * pca_dot_pca)

class dataCompress_nD(object):
    """
    Class for dimensional reduction of n-D histogram data using
    multilinear PCA through a Tucker decomposition
    """

    def __init__(self, dataMat, histBins, simDesign, tol=1e-5,
                 user_dim=5, user_dim2=None):
        """
        :param dataMat: Matrix of simulation data [nbins x nbins2 x nsims]
        :param histBins: Coordinates of bins
        :param simDesign: Coordinates of simulations
        :param tol: PCA tolerance for singular values [default = 1e-5]
        :param user_dim: Number of elements in low-rank form [default = 5]
        :param user_dim2: Number of elements in 2nd dimension of low-rank form [default = None]

        :return pixMean: Mean of each tensor bin
        :return pixStd: Standard deviation of each tensor bin
        """
        self.dataMat = dataMat
        self.unitMat = None
        self.histBins = histBins
        self.simDesign = simDesign
        self.tol = tol
        self.user_dim = user_dim
        if user_dim2 is None:
            self.user_dim2 = self.user_dim
        else:
            self.user_dim2 = user_dim2
        self.core = None
        self.unitCore = None
        self.coreStd = None
        self.factors = None
        self.pixMean = np.mean(dataMat,axis=-1)
        self.pixStd = np.std(dataMat,axis=-1)

    def center(self):
        """
        Convert each tensor bin to zero-mean

        :return unitMat: zero-mean tensor bins
        """
        self.unitMat = self.dataMat.copy() - self.pixMean[:,:,None]

    def basisCompute(self):
        """
        PCA compression of simulation data.

        :return pca_basis: Matrix with columns as orthogonal basis vectors
        :return pca_weights: Coefficients of basis vectors at simulation points
        """
        X = tl.tensor(self.unitMat)

        self.core, self.factors = tld.partial_tucker(X[:,:,:], modes = [0,1], tol=self.tol,
                                                     ranks = [self.user_dim, self.user_dim2])
        self.core = tlb.to_numpy(self.core)

    def unitVar(self):
        """
        Convert PCA basis pixel to unit variance

        :return unitCore: unit variance core tensor
        """
        self.coreStd = np.std(self.core.copy(),axis=-1)
        self.unitCore = self.core.copy() / self.coreStd[:,:,None]

    def rotate2full(self, mat, error=False):
        """
        Rotate a prediction back into full-rank space (i.e. bin space)

        :param mat: Tensor in low-rank space
        :param error: Is this a tensor of uncertainties? [default = False]

        :return full_rank: Full rank prediction
        """
        if self.unitCore is not None:
            tmp_mat = mat * self.coreStd
        else:
            tmp_mat = mat

        full_rank = tlb.to_numpy(tl.tenalg.multi_mode_dot(tl.tensor(tmp_mat), self.factors,
                                                          modes=[0, 1], transpose=False))

        if self.unitMat is not None:
            full_rank += self.pixMean
            if error:
                full_rank -= self.pixMean

        return full_rank

    def match(self, test_dataMat=None):
        """
        Return match of un-compressed data against the original data

        :return match: match vector across all simulations
        """
        data_dot_pca = np.array([np.dot(self.dataMat[:,:,ii].flatten(),
                                        self.rotate2full(self.unitCore[:,:,ii]).flatten())
                                 for ii in range(self.dataMat.shape[-1])])
        data_dot_data = np.array([np.dot(self.dataMat[:,:,ii].flatten(),
                                         self.dataMat[:,:,ii].flatten())
                                  for ii in range(self.dataMat.shape[-1])])
        pca_dot_pca = np.array([np.dot(self.rotate2full(self.unitCore[:,:,ii]).flatten(),
                                       self.rotate2full(self.unitCore[:,:,ii]).flatten())
                                for ii in range(self.dataMat.shape[-1])])
        return data_dot_pca / np.sqrt(data_dot_data * pca_dot_pca)
