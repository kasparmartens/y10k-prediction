import sys
sys.path.append("/home/kaspar")
from y10k_prediction.imports import *

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
