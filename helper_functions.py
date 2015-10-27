import scipy as sp
import pylab as pl
import scipy.stats as st
import numpy as np
import pandas as pd

def summarise_Rsq(Y, Ypred):
    P = Y.shape[1]
    Rsq = np.zeros(P)
    for j in range(P):
        Rsq[j] = sp.corrcoef(Ypred[:, j], Y[:, j])[0,1]**2
    return np.row_stack((Rsq))
