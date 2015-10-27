import scipy as sp
import pylab as pl
import scipy.stats as st
import numpy as np
import pandas as pd

import limix.modules.varianceDecomposition as var
import limix.modules.qtl as qtl
import limix.io.data as data
import limix.io.genotype_reader as gr
import limix.io.phenotype_reader as phr
import limix.io.data_util as data_util
import limix.utils.preprocess as preprocess
from limix.utils.plot import *
from limix.stats.geno_summary import *
from sklearn import linear_model


def get_BLUPs(Y, K, Itrain=None, Itest=None):
    if Itrain is None:
        Itrain = sp.ones(Y.shape[0], dtype=bool)
        Itest = sp.ones(Y.shape[0], dtype=bool)
    m = var.VarianceDecomposition(Y[Itrain])
    m.setTestSampleSize(Itest.sum())
    m.addFixedEffect()
    m.addRandomEffect(K=K[Itrain, :][:, Itrain], Kcross=K[Itrain, :][:, Itest])
    m.addRandomEffect(is_noise=True)
    m.optimize()
    blups = m.predictPhenos()
    return blups

def get_BLUPs_with_confidence(Y, K, Itrain, Itest):
    m = var.VarianceDecomposition(Y[Itrain])
    m.setTestSampleSize(Itest.sum())
    m.addFixedEffect()
    m.addRandomEffect(K=K[Itrain, :][:, Itrain], Kcross=K[Itrain, :][:, Itest])
    m.addRandomEffect(is_noise=True)
    m.optimize()
    blups = m.predictPhenos()
    varcomps = m.getVarianceComps().ravel()
    Sigma = varcomps[0]*K + varcomps[1]*sp.eye(Y.shape[0])
    Sigma_train_inv = np.linalg.inv(Sigma[Itrain, :][:, Itrain])
    var_predictive = Sigma[Itest, :][:, Itest] - sp.dot(sp.dot(Sigma[Itest, :][:, Itrain], Sigma_train_inv), Sigma[Itrain, :][:, Itest])
    return {"pred": blups, "predictive_sd": np.sqrt(np.diag(var_predictive))}
