import numpy as np
import sklearn
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.base import RegressorMixin, BaseEstimator
from .hamiltonians import _orbs_offsets, _atom_blocks_idx
from time import time
import scipy

class SASplitter:
    """ CV splitter that takes into account the presence of "L blocks"
    associated with symmetry-adapted regression. Basically, you can trick conventional
    regression schemes to work on symmetry-adapted data y^M_L(A_i) by having the (2L+1)
    angular channels "unrolled" into a flat array. Then however splitting of train/test
    or cross validation must not "cut" across the M block. This takes care of that.
    """
    def __init__(self, L, cv=2):
        self.L = L
        self.cv = cv
        self.n_splits = cv

    def split(self, X, y, groups=None):

        ntrain = X.shape[0]
        if ntrain % (2*self.L+1) != 0:
            raise ValueError("Size of training data is inconsistent with the L value")
        ntrain = ntrain // (2*self.L+1)
        nbatch = (2*self.L+1)*(ntrain//self.n_splits)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for n in range(self.n_splits):
            itest = idx[n*nbatch:(n+1)*nbatch]
            itrain = np.concatenate([idx[:n*nbatch], idx[(n+1)*nbatch:]])
            yield itrain, itest

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    

class SARidge(Ridge):
    """ Symmetry-adapted ridge regression class """

    def __init__(self, L, alpha=1, alphas=None, cv=2, solver='auto',
                 fit_intercept=False, scoring='neg_root_mean_squared_error'):
        self.L = L
        # L>0 components have zero mean by symmetry
        if L>0:
            fit_intercept = False
        self.cv = SASplitter(L, cv)
        self.alphas = alphas
        self.cv_stats = None
        self.scoring = scoring
        self.solver = solver
        super(SARidge, self).__init__(alpha=alpha, fit_intercept=fit_intercept, solver=solver)

    def fit(self, Xm, Ym, X0=None):
        # this expects properties in the form [i, m] and features in the form [i, q, m]
        # in order to train a SA-GPR model the m indices have to be moved and merged with the i

        Xm_flat = np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1]))
        Ym_flat = Ym.flatten()
        if self.alphas is not None:
            # determines alpha by grid search
            rcv = Ridge(fit_intercept=self.fit_intercept)
            gscv = GridSearchCV(rcv, dict(alpha=self.alphas), cv=self.cv, scoring=self.scoring)
            gscv.fit(Xm_flat, Ym_flat)
            self.cv_stats = gscv.cv_results_
            self.alpha = gscv.best_params_["alpha"]

        super(SARidge, self).fit(Xm_flat, Ym_flat)
    def predict(self, Xm, X0=None):

        Y = super(SARidge, self).predict(np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1])))
        return Y.reshape((-1, 2*self.L+1))
    
    
class Fock_regression():
    def __init__(self, orbs, *args, **kwargs):
        self._orbs = orbs
        _, self._eldict = _orbs_offset(orbs)
        self._args = args
        self._kwargs = kwargs
        self.model_template = SARidge
        
    def fit(self, feats, fock_bc, slices=None, progress=None):
        self._models = {}
        self._cv_stats = {}
        blocks = []
        block_idx = []
        for idx_fock, block_fock in fock_bc:
            block_type1, ai, ni, li, aj, nj, lj, L1 = tuple(idx_fock)
            for idx_feats, block_feats in feats:
                block_type2, L2, nu, sigma, sp_i, sp_j = tuple(idx_feats)
                tgt = block_feats.values
                
                if not(block_type1==block_type2 and L1==L2 and ai==sp_i and aj==sp_j):
                    continue
                    block_idx.append(tuple(idx_fock))
                    block_data = self._model_template(L1, *self._args, **self._kwargs)
                    block_samples = block_fock.samples
                    
                    
                    newblock = TensorBlock(
                        values=block_data,
                        samples=block_samples,
                        components=[Labels(
                            ["mu"], np.asarray(range(-L1, L1 + 1), dtype=np.int32).reshape(-1, 1)
                        )],
                        properties= Labels("values", np.asarray(X_idx[block_idx], dtype=np.int32))

                    blocks.append(newblock) 
                
        self._models = TensorMap(Labels(fock_bc.keys.names, np.asarray(block_idx, dtype=np.int32)), blocks)
                        

                
                        
            


            