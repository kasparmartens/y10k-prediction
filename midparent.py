import patsy
import scipy as sp
import numpy as np
from sklearn import linear_model

def midparent_predictions(Y, parent1, parent2, Itrain=None, Itest=None):
    if Itrain is None:
    	Itrain = sp.ones(Y.shape[0], dtype=bool)
    	Itest = sp.ones(Y.shape[0], dtype=bool)
    modelmatrix = patsy.dmatrix("0 + parent1 + parent2")
    Rsquared = sp.zeros((Y.shape[1]))
    pred = sp.zeros_like(Y[Itest, :])
    for j in range(Y.shape[1]):
        lm = linear_model.LinearRegression()
        lm.fit(modelmatrix[Itrain, :], Y[Itrain, j])
        pred[:, j] = lm.predict(modelmatrix[Itest, :])
        Rsquared[j] = sp.corrcoef(pred[:, j], Y[Itest, j])[0,1]**2
    return np.row_stack((Rsquared)), pred
