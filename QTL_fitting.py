import sys
sys.path.append("/home/kaspar")
from y10k_prediction.imports import *
from y10k_prediction.LMM import *

def get_predictions_iid(y, covs, Itrain, Itest):
    lm = linear_model.LinearRegression()
    lm.fit(covs[Itrain, :], y[Itrain])
    pred = lm.predict(covs[Itest, :])
    return pred

def get_predictions_iid_with_weights(y, covs, Itrain, Itest):
    lm = linear_model.LinearRegression()
    lm.fit(covs[Itrain, :], y[Itrain])
    pred = lm.predict(covs[Itest, :])
    coefs = lm.coef_
    residuals = (y[Itrain] - lm.predict(covs[Itrain, :])).ravel()
    sigma2 = sum(residuals**2) / (Itrain.sum() - covs.shape[1])
    return pred, coefs, sigma2

def QTL_iid_predictions(Y, which_col, snps, Itrain, Itest, pred_nQTLs, maxiter=10, pvalue_threshold=1, n_folds = 4, verbose=False):
    # first, select ordering of covariates using Itrain 
    obj = add_QTLs_conditional(Y, which_col, snps, "iid", Itrain, maxiter, pvalue_threshold, allow_interactions=False, conditional=False, verbose=verbose)
    y = Y[:, which_col]
    covs = obj["covs"]
    iadded = np.array(obj["iadded"])
    interactions = np.array(obj["interactions"])
    pvadded = obj["pvadded"]

    # select optimal number of covariates
    ysub = y[Itrain]
    covs_sub = covs[Itrain, :]

    N = ysub.shape[0]
    m = len(pred_nQTLs)
    pred = sp.zeros((N, m))
    r = sp.random.permutation(N)
    Icv = sp.floor(((sp.ones((N))*n_folds)*r)/N)
    for i in range(n_folds):
        Itrain_inner = (Icv != i)
        Itest_inner = (Icv == i)
        for k in range(m):
            pred[Itest_inner, k] = get_predictions_iid(ysub, covs_sub[:, 0:(pred_nQTLs[k]+1)], Itrain_inner, Itest_inner)
        
    Rsquared = sp.zeros(m)
    for k in range(m):
        Rsquared[k] = sp.corrcoef(pred[:, k], ysub)[0, 1]**2
    best_col = np.argmax(Rsquared)
    best_pred = pred[:, best_col]
    n_features = (pred_nQTLs[best_col]+1)
    best_covs = covs[:, 0:n_features]

    final_predictions = get_predictions_iid(y, best_covs, Itrain, Itest)
    Rsq_final = sp.corrcoef(final_predictions, y[Itest])[0, 1]**2

    return {"initial_Rsquared": Rsquared, 
            "initial_pred": pred, 
            "covs": covs, 
            "best_covs": best_covs, 
            "iadded": iadded, 
            "interactions": interactions, 
            "pvadded": pvadded, 
            "pred": final_predictions, 
            "Rsquared": Rsq_final}

def QTL_iid_predictions_exploration(Y, which_col, snps, Itrain, Itest, pred_nQTLs, maxiter=10, pvalue_threshold=1, n_folds=4, verbose=False):
    # first, select ordering of covariates using Itrain 
    obj = add_QTLs_conditional(Y, which_col, snps, "iid", Itrain, maxiter, pvalue_threshold, allow_interactions=False, conditional=False, verbose=verbose)
    y = Y[:, which_col]
    covs = obj["covs"]
    iadded = np.array(obj["iadded"])
    interactions = np.array(obj["interactions"])
    pvadded = obj["pvadded"]

    # select optimal number of covariates
    ysub = y[Itrain]
    covs_sub = covs[Itrain, :]

    N = ysub.shape[0]
    m = len(pred_nQTLs)
    pred = sp.zeros((N, m))
    r = sp.random.permutation(N)
    Icv = sp.floor(((sp.ones((N))*n_folds)*r)/N)
    for i in range(n_folds):
        Itrain_inner = (Icv != i)
        Itest_inner = (Icv == i)
        for k in range(m):
            pred[Itest_inner, k] = get_predictions_iid(ysub, covs_sub[:, 0:(pred_nQTLs[k]+1)], Itrain_inner, Itest_inner)
    Rsquared = sp.zeros(m)
    for k in range(m):
        Rsquared[k] = sp.corrcoef(pred[:, k], ysub)[0, 1]**2
    best_col = np.argmax(Rsquared)
    n_features = (pred_nQTLs[best_col]+1)
    best_covs = covs[:, 0:n_features]
    final_predictions = get_predictions_iid(y, best_covs, Itrain, Itest)
    Rsq_final = sp.corrcoef(final_predictions, y[Itest])[0, 1]**2
    
    
    pred_test = sp.zeros((Itest.sum(), m))
    pred_train = sp.zeros((Itrain.sum(), m))
    Rsquared_test = sp.zeros(m)
    Rsquared_train = sp.zeros(m)
    for k in range(m):
        pred_test[:, k] = get_predictions_iid(y, covs[:, 0:(pred_nQTLs[k]+1)], Itrain, Itest)
        pred_train[:, k] = get_predictions_iid(y, covs[:, 0:(pred_nQTLs[k]+1)], Itrain, Itrain)
        Rsquared_test[k] = sp.corrcoef(pred_test[:, k], y[Itest])[0, 1]**2
        Rsquared_train[k] = sp.corrcoef(pred_train[:, k], y[Itrain])[0, 1]**2
    
    return {"initial_Rsquared": Rsquared, 
            "initial_pred": pred_test, 
            "covs": covs, 
            "best_covs": best_covs, 
            "iadded": iadded, 
            "interactions": interactions, 
            "pvadded": pvadded, 
            "pred": final_predictions, 
            "Rsquared": Rsq_final, 
            "Rsq_test": Rsquared_test, 
            "Rsq_train": Rsquared_train}
