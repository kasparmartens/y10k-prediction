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

def get_heritability(Y, K):
    m = var.VarianceDecomposition(Y)
    m.addFixedEffect()
    m.addRandomEffect(K=K, normalize=False)
    m.addRandomEffect(is_noise=True)
    m.optimize()
    varcomps = m.getVarianceComps().ravel()
    kinship = varcomps[0]
    noise = varcomps[1]
    h2 = kinship / (noise + kinship)
    return h2

def get_heritability_with_SE(Y, K):
    m = var.VarianceDecomposition(Y)
    m.addFixedEffect()
    m.addRandomEffect(K=K, normalize=False)
    m.addRandomEffect(is_noise=True)
    m.optimize()
    varcomps = m.getVarianceComps().ravel()
    kinship = varcomps[0]
    noise = varcomps[1]
    h2 = kinship / (noise + kinship)
    SE = getVarCompSE_LMM(m)
    return {"h2": h2, "SE": SE}



def getVarCompSE_LMM(m):
    RV=sp.zeros(m.n_randEffs)
    for term_i in range(m.n_randEffs):
        RV[term_i] = m.getTraitCovarStdErrors(term_i)
    return RV


def calculate_kinship(geno_matrix):
    X = sp.array(geno_matrix)
    snp_means = X.mean(axis=0)
    Z = (X - snp_means[np.newaxis, :]) 
    snp_sd = Z.std(axis=0) + 1.0e-8
    Z = Z / snp_sd[np.newaxis, :]
    W = Z.dot(Z.T)
    out = W / W.diagonal().mean()
    return out

def calculate_kinship_mod(geno_matrix):
    X = sp.array(geno_matrix)
    snp_means = X.mean(axis=0)
    Z = (X - snp_means[np.newaxis, :]) 
    snp_sd = Z.std(axis=0) + 1.0e-8
    Z = Z / snp_sd[np.newaxis, :]
    W = Z.dot(Z.T)
    return W


def find_next_term(snps, y, K, Itrain, covs, interactions, iadded, allow_interactions, pvalue_threshold=1, verbose=True):
    indexes = []
    pv = []
    new_variable = []
    M = snps.shape[1]
    itested = [0]
    X = snps
    # if interactions are allowed, search for those
    if allow_interactions:
        for i in range(1, covs.shape[1]):
            if interactions[i] == 0:
                itested.append(iadded[i])
                column = covs[:, i]
                temp = (column * snps.T).T
                X = np.column_stack((X, temp))
                    
    if K == "iid":
        lmm = qtl.test_lm(X[Itrain, :], y[Itrain], covs = covs[Itrain, :])
    else:
        lmm = qtl.test_lmm(X[Itrain, :], y[Itrain], K=K[Itrain, :][:, Itrain], searchDelta=False, covs=covs[Itrain, :])

    pv = lmm.getPv().ravel()
    which_snp = pv.argmin()
    index = which_snp % M
    pvalue = pv[which_snp]
    new_variable = X[:, which_snp]
    interaction_id = (which_snp / M)
    
    return {"index": index, "interaction": itested[interaction_id], "pvalue": pvalue, "new_variable": new_variable}

def add_QTLs(y, snps, K, Itrain, maxiter=10, pvalue_threshold = 1, allow_interactions=True, update_kinship=False, verbose=True):
    covs = np.row_stack((np.ones((len(y)))))
    iadded = [0]
    pvadded = [1]
    interactions = [0]
    for k in range(1, maxiter + 1):
        if update_kinship:
            indexes = find_correlated_snp_indexes(snps, iadded)
            subset_snps = leave_out_snps(snps, indexes)
            K = calculate_kinship(subset_snps)
        obj = find_next_term(snps, y, K, Itrain, covs, interactions, iadded, allow_interactions, pvalue_threshold, verbose=verbose)

        if obj["pvalue"] < pvalue_threshold:
            print obj["index"]
            iadded.append(obj["index"])
            pvadded.append(obj["pvalue"])
            interactions.append(obj["interaction"])
            covs = np.column_stack((covs, obj["new_variable"]))
        else:
            break
    
    out = {}
    out["iadded"] = iadded
    out["pvadded"] = pvadded
    out["interactions"] = interactions
    out["covs"] = covs
    return out

def add_QTLs_conditional(Y, which_col, X, K, Itrain, maxiter=10, pvalue_threshold = 1, allow_interactions=True, conditional=True, verbose=True):
    y = Y[:, which_col]
    covs = np.ones((Y.shape[0], 1))
    iadded = [0]
    interactions = [0]
    pvadded = [1]
    if conditional:
        for j in range(Y.shape[1]):
            if j != which_col:
                covs = np.column_stack((covs, Y[:, j]))
                iadded.append(X.shape[1] + j)
                interactions.append(-1)
                pvadded.append(1)
    for k in range(1, maxiter + 1):
        obj = find_next_term(X, y, K, Itrain, covs, interactions, iadded, allow_interactions, pvalue_threshold, verbose=verbose)

        if obj["pvalue"] < pvalue_threshold:
            iadded.append(obj["index"])
            pvadded.append(obj["pvalue"])
            interactions.append(obj["interaction"])
            covs = np.column_stack((covs, obj["new_variable"]))
        else:
            break
    out = {}
    out["iadded"] = iadded
    out["pvadded"] = pvadded
    out["interactions"] = interactions
    out["covs"] = covs
    return out


def conditional_predictions(Y, which_col, snps, K, Itrain, Itest, pred_nQTLs, maxiter=10, pvalue_threshold=1, allow_interactions=True, conditional=True, n_folds = 4, return_decomposition=True, verbose=False):
    # first, select ordering of covariates using Itrain 
    obj = add_QTLs_conditional(Y, which_col, snps, K, Itrain, maxiter, pvalue_threshold, allow_interactions, conditional, verbose)
    y = Y[:, which_col]
    covs = obj["covs"]
    iadded = np.array(obj["iadded"])
    interactions = np.array(obj["interactions"])
    pvadded = obj["pvadded"]
    if conditional:
        covs_additional = Y.shape[1]
    else:
        covs_additional = 1
    # select optimal number of covariates
    ysub = y[Itrain]
    covs_sub = covs[Itrain, :]
    Ksub = K[Itrain, :][:, Itrain]
    N = ysub.shape[0]
    m = len(pred_nQTLs)
    pred = sp.zeros((N, m))
    r = sp.random.permutation(N)
    Icv = sp.floor(((sp.ones((N))*n_folds)*r)/N)
    for i in range(n_folds):
        Itrain_inner = (Icv != i)
        Itest_inner = (Icv == i)
        for k in range(m):
            pred[Itest_inner, k] = get_predictions(ysub, Ksub, covs_sub[:, 0:(pred_nQTLs[k]+covs_additional)], Itrain_inner, Itest_inner)
        
    Rsquared = sp.zeros(m)
    for k in range(m):
        Rsquared[k] = sp.corrcoef(pred[:, k], ysub)[0, 1]**2
    best_col = np.argmax(Rsquared)
    best_pred = pred[:, best_col]
    n_features = (pred_nQTLs[best_col]+covs_additional)
    best_covs = covs[:, 0:n_features]
    iadded = iadded[0:n_features]
    interactions = interactions[0:n_features]

    final_predictions = get_predictions(y, K, best_covs, Itrain, Itest)
    Rsq_LMMP = sp.corrcoef(final_predictions, y[Itest])[0, 1]**2

    if return_decomposition:
        # without phenotypes (LMM)
        pred1 = get_predictions(y, K, best_covs[:, interactions != -1], Itrain, Itest)
        Rsq_LMM_dom_int = sp.corrcoef(pred1, y[Itest])[0, 1]**2
        # LMM + dominance
        pred1 = get_predictions(y, K, best_covs[:, ((interactions == 0) | (interactions == iadded))], Itrain, Itest)
        Rsq_LMM_dom = sp.corrcoef(pred1, y[Itest])[0, 1]**2
        # LMM
        pred1 = get_predictions(y, K, best_covs[:, (interactions == 0)], Itrain, Itest)
        Rsq_LMM = sp.corrcoef(pred1, y[Itest])[0, 1]**2
        Rsq = np.array([Rsq_LMMP, Rsq_LMM_dom_int, Rsq_LMM_dom, Rsq_LMM])
    else:
        Rsq = Rsq_LMMP

    return {"initial_Rsquared": Rsquared, 
            "initial_pred": pred, 
            "covs": covs, 
            "best_covs": best_covs, 
            "iadded": iadded, 
            "interactions": interactions, 
            "pvadded": pvadded, 
            "pred": final_predictions, 
            "Rsquared": Rsq}


def get_predictions(y, K, covs, Itrain, Itest):

    model = var.VarianceDecomposition(y[Itrain]) # set phenotype matrix Y [N,P]
    model.setTestSampleSize(Itest.sum())
    model.addFixedEffect(F=covs[Itrain, :], Ftest=covs[Itest, :]) # set fixed effects
    if K != "iid":
        model.addRandomEffect(K=K[Itrain, :][:, Itrain], Kcross=K[Itrain, :][:, Itest])
    model.addRandomEffect(is_noise=True) # set random effect for measurement error
    model.optimize()
    predictions = model.predictPhenos().ravel()
    return predictions


def get_prediction_accuracy(y, snps, K, covs, iadded, Itrain, Itest, update_kinship=False):
    m = covs.shape[1]
    Rsquared = sp.zeros(m)
    pred = sp.zeros((Itest.sum(), m))
    # Loop over columns of covs matrix, include gradually columns 0:(k+1)
    for k in range(m):
        pred[:, k] = get_predictions(y, K, covs[:, 0:(k+1)], iadded, Itrain, Itest, update_kinship)
        Rsquared[k] = sp.corrcoef(pred[:, k], y[Itest])[0, 1]**2
    best_col = np.argmax(Rsquared)
    best_pred = pred[:, best_col]
    return {"Rsquared": Rsquared, "best_pred": best_pred}


def summarise_QTLs(res, environments):
    pieces = []
    for i in range(len(res)):
        temp = res[i]
        pieces.append(pd.DataFrame({"iadded": temp["iadded"], "interactions": temp["interactions"], "pvalue": temp["pvadded"], "environment": environments[i]}))
    df = pd.concat(pieces)
    return df

def summarise_Rsq(Y, Ypred):
    P = Y.shape[1]
    Rsq = np.zeros(P)
    for j in range(P):
        Rsq[j] = sp.corrcoef(Ypred[:, j], Y[:, j])[0,1]**2
    return np.row_stack((Rsq))

def summarise_Rsq_mod(Y, obj):
    P = Y.shape[1]
    Rsq = np.zeros(P)
    for j in range(P):
        ypred = obj[j]
        Rsq[j] = sp.corrcoef(ypred.ravel(), Y[:, j:j+1].ravel())[0,1]**2
    return np.row_stack((Rsq))

def summarise_Rsq_ver2(Ypred_list, Y, environments, pred_depths):
    Rsq = sp.zeros((Ypred_list[0].shape[1], Y.shape[1]))
    for j in range(len(Ypred_list)):
        Ypred = Ypred_list[j]
        for i in range(Ypred.shape[1]):
            Rsq[i, j] = sp.corrcoef(Ypred[:, i], Y[:, j])[0,1]**2
    df = pd.DataFrame(Rsq, columns = environments)
    return df

def QTLs_LOO_exploration(Y, which_col, snps, Itrain_list, Itest_list, maxiter=10, pvalue_threshold=1, n_folds=4, verbose=False):
    pred_nQTLs = range(maxiter+1)
    Ntotal = len(Itrain_list)
    pred_test = sp.zeros(Ntotal)
    ytest = sp.zeros(Ntotal)
    Rsq_train = sp.zeros(Ntotal)
    for i in range(Ntotal):
        Itest = Itest_list[i]
        Itrain = Itrain_list[i]
        
        obj = add_QTLs_conditional(Y, which_col, snps, "iid", Itrain, maxiter, pvalue_threshold, allow_interactions=False, conditional=False, verbose=verbose)
        y = Y[:, which_col]
        covs = obj["covs"]
        iadded = np.array(obj["iadded"])
        interactions = np.array(obj["interactions"])
        pvadded = obj["pvadded"]

        pred_train, weights = get_predictions_iid_with_weights(y, covs[:, 0:(maxiter+1)], Itrain, Itrain)
        Rsq_train[i] = sp.corrcoef(pred_train, y[Itrain])[0, 1]**2
        
        pred_test[i] = get_predictions_iid(y, covs[:, 0:(maxiter+1)], Itrain, Itest)
        ytest[i] = y[Itest]
        temp = pd.DataFrame({"iadded": iadded[1:len(iadded)], "weights": weights[1:len(weights)], "iter": i})
        if i==0:
            qtls = temp
        else:
            qtls = pd.concat((qtls, temp))
        
    Rsq_test = sp.corrcoef(pred_test, ytest)[0, 1]**2
    return {"Rsq_test": Rsq_test, 
            "Rsq_train": Rsq_train, 
            "pred": pred_test,
            "obs": ytest, 
            "QTLs": qtls}

