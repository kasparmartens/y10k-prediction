import sys
sys.path.append("/home/kaspar")
from y10k_prediction.imports import *

def summarise_Rsq(Y, Ypred):
    P = Y.shape[1]
    Rsq = np.zeros(P)
    for j in range(P):
        Rsq[j] = sp.corrcoef(Ypred[:, j], Y[:, j])[0,1]**2
    return np.row_stack((Rsq))
