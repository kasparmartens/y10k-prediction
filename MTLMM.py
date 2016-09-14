import sys
sys.path.append("/home/kaspar")
from y10k_prediction.imports import *

def fit_simple_MT_LMM(Y, K, Itrain):
    m = var.VarianceDecomposition(Y[Itrain, :])
    m.setTestSampleSize(Y.shape[0])
    m.addFixedEffect()
    m.addRandomEffect(K=K[Itrain, :][:, Itrain])
    m.addRandomEffect(is_noise=True)
    m.optimize()
    return m


def build_MT_LMM(Y, K, covmat_list, Itrain):
    m = var.VarianceDecomposition(Y[Itrain, :])
    m.setTestSampleSize(Y.shape[0])
    for j in range(Y.shape[1]):
        covmat = covmat_list[j]
        A = sp.zeros((1, Y.shape[1]))
        A[0, j] = 1
        m.addFixedEffect(F=covmat[Itrain, :], Ftest=covmat, A=A)
    m.addRandomEffect(K=K[Itrain, :][:, Itrain])
    m.addRandomEffect(is_noise=True)
    m.optimize()
    return m



def MTLMM_conditional_pred(Y, K, Itrain, Itest, m):
    C_g = m.getTraitCovar(0)
    C_n = m.getTraitCovar(1)
    P = Y.shape[1]

    Sigma = np.kron(C_g, K) + np.kron(C_n, np.eye(K.shape[0]))
    
    oldind, newind, testind = calc_indexes(Itrain, Itest, P=Y.shape[1], which_col=0)
    A = Sigma[oldind, :][:, oldind]
    A_inv = np.linalg.inv(A)
    fixed_eff_pred = m.predictPhenos()
    Yres = Y - fixed_eff_pred
    
    Ypred = sp.zeros((Itest.sum(), Y.shape[1]))
    Ymarginalpred = sp.zeros((Itest.sum(), Y.shape[1]))

    for j in range(P):
        obj = helper_conditional_pred(Yres, j, Itrain, Itest, Sigma, A_inv)
        Ypred[:, j:j+1] = obj["pred1"]
        Ymarginalpred[:, j:j+1] = obj["pred2"]
    
    res_old = Yres.reshape((-1, 1), order="F")[oldind, :]
    res_new = Yres.reshape((-1, 1), order="F")[newind, :]
    
    Ypred += fixed_eff_pred[Itest, :]
    Ymarginalpred += fixed_eff_pred[Itest, :]
    
    return Ypred, Ymarginalpred


def helper_conditional_pred(Yres, which_col, Itrain, Itest, Sigma, A_inv):
    oldind, newind, testind = calc_indexes(Itrain, Itest, P=Yres.shape[1], which_col=which_col)
    
    Knew = Sigma[testind, :][:, newind]
    Kold = Sigma[testind, :][:, oldind]
    temp1 = np.dot(Kold, A_inv)

    B = Sigma[oldind, :][:, newind]
    C = Sigma[newind, :][:, oldind]
    D = Sigma[newind, :][:, newind]

    res_old = Yres.reshape((-1, 1), order="F")[oldind, :]
    res_new = Yres.reshape((-1, 1), order="F")[newind, :]
    
    pred1, pred2 = matrix_helper_conditional_pred(B, C, D, A_inv, res_old, res_new, Kold, Knew, temp1)
    return {"pred1": pred1, "pred2": pred2}

def matrix_helper_conditional_pred(B, C, D, A_inv, res_old, res_new, Kold, Knew, temp1):
    E = np.linalg.inv(D - np.dot(np.dot(C, A_inv), B))

    temp2 = np.dot(np.dot(temp1, B), E)
    temp3 = np.dot(C, np.dot(A_inv, res_old))
    temp4 = np.dot(Knew, E)

    comp1 = np.dot(temp1, res_old)
    comp2 = np.dot(temp2, temp3)
    comp3 = np.dot(temp4, temp3)
    comp4 = np.dot(temp2, res_new)
    comp5 = np.dot(temp4, res_new)

    pred = comp1 + comp2 - comp3 - comp4 + comp5
    return pred, comp1

def calc_indexes(Itrain, Itest, P, which_col=0):
    oldind = np.array([], dtype=bool)
    newind = np.array([], dtype=bool)
    testind = np.array([], dtype=bool)
    for i in range(P):
        oldind = np.concatenate((oldind, Itrain))
        if i == which_col:
            newind = np.concatenate((newind, np.zeros(len(Itest), dtype=bool)))
            testind = np.concatenate((testind, Itest))
        else:
            newind = np.concatenate((newind, Itest))
            testind = np.concatenate((testind, np.zeros(len(Itest), dtype=bool)))
    return oldind, newind, testind

# def conditional_predictions(Y, model, R, Itrain, Itest, which_col = 0):
#     allind = Itrain + Itest
#     Y = Y[allind, :]
#     R = R[allind, :][:, allind]
#     Itest = Itest[allind]
#     Itrain = Itrain[allind]
    
#     fixed_effect_mat = model.predictPhenos()[allind, :]
#     residuals_mat = Y - fixed_effect_mat
#     residuals = residuals_mat.reshape(-1, 1, order="F")
    
#     # ind is the indicator for predicting vec(Y[Itest, which_col]). 
#     ind = np.array([], dtype=bool)
#     for i in range(Y.shape[1]):
#         if i == which_col:
#             ind = np.concatenate((ind, Itest))
#         else:
#             ind = np.concatenate((ind, np.zeros(Y.shape[0], dtype=bool)))

#     C_g = model.getTraitCovar(0)
#     C_n = model.getTraitCovar(1)
#     Sigma_hat = np.kron(C_g, R) + np.kron(C_n, np.eye(Y.shape[0]))

#     Sigma11 = Sigma_hat[ind, :][:, ind]
#     Sigma12 = Sigma_hat[ind, :][:, ~ind]
#     Sigma22 = Sigma_hat[~ind, :][:, ~ind]
#     Sigma22inv = np.linalg.inv(Sigma22)
    
#     predictions = fixed_effect_mat[Itest, which_col:(which_col+1)] + np.dot(np.dot(Sigma12, Sigma22inv), residuals[~ind])
#     return predictions.ravel()
