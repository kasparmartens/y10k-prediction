import sys
import scipy as sp
import pylab as pl
import scipy.stats as st
import numpy as np
import pandas as pd

def get_Itrain_Itest(N, proportion = 0.8, seed=0):
    sp.random.seed(seed)
    n1 = int(sp.floor(N*proportion))
    n2 = int(N - n1)
    selected_indexes = sp.concatenate((sp.ones(n1, dtype=bool), sp.zeros(n2, dtype=bool)))
    sp.random.shuffle(selected_indexes)
    Itrain = selected_indexes
    Itest = ~selected_indexes
    return Itrain, Itest


def get_CV_ind(Nobs, n_folds=5):
    r = sp.random.permutation(Nobs)
    Icv = sp.floor(((sp.ones((Nobs))*n_folds)*r)/Nobs)
    return Icv, n_folds


def select_subset(x, n):
    out = sp.zeros_like(x, dtype=bool)
    out[np.random.choice(np.where(x)[0], size=n, replace=False)] = True
    return out

def get_4_squares(parent1, parent2):
    n_folds = 2
    levels1 = np.unique(parent1)
    levels2 = np.unique(parent2)
    N1 = len(levels1)
    N2 = len(levels2)
    r1 = sp.random.permutation(N1)
    r2 = sp.random.permutation(N2)
    Icv1 = sp.floor(((sp.ones((N1))*n_folds)*r1)/N1)
    Icv2 = sp.floor(((sp.ones((N2))*n_folds)*r2)/N2)

    train_parents1 = levels1[Icv1 != 0]
    train_parents2 = levels2[Icv2 != 0]
    test_parents1 = levels1[Icv1 == 0]
    test_parents2 = levels2[Icv2 == 0]

    train_ind1 = np.array([e in train_parents1 for e in parent1], dtype=bool)
    train_ind2 = np.array([e in train_parents2 for e in parent2], dtype=bool)
    test_ind1 = np.array([e in test_parents1 for e in parent1], dtype=bool)
    test_ind2 = np.array([e in test_parents2 for e in parent2], dtype=bool)

    Itest = test_ind1 & test_ind2
    
    Itrain_distant = train_ind1 & train_ind2
    Itrain_close1 = (train_ind1 & test_ind2)
    Itrain_close2 = (train_ind2 & test_ind1)
    Itrain_close = select_subset(Itrain_close1 | Itrain_close2, Itest.sum())

    return Itest, Itrain_distant, Itrain_close1, Itrain_close2, Itrain_close

def get_4foldCV(parent1, parent2):
    ind1, ind4, ind2, ind3, temp = get_4_squares(parent1, parent2)
    all = ind1 | ind2 | ind3 | ind4
    Itest = [ind1, ind2, ind3, ind4]
    Itrain = [all - ind1, all - ind2, all - ind3, all - ind4]
    return Itest, Itrain
    
def get_4foldCV_close_and_distant(parent1, parent2):
    ind1, ind4, ind2, ind3, temp = get_4_squares(parent1, parent2)
    n = ind1.sum()
    all = ind1 | ind2 | ind3 | ind4
    Itest = [ind1, ind2, ind3, ind4]
    Idistant = [ind4, ind3, ind2, ind1]
    Iclose = [select_subset(ind2|ind3, n), select_subset(ind1|ind4, n), select_subset(ind1|ind4, n), select_subset(ind2|ind3, n)]
    return Itest, Idistant, Iclose

