import sys
sys.path.append("/home/kaspar")
from y10k_prediction.imports import *

def get_data_without_parents(file_name, environments=None):

    geno_reader  = gr.genotype_reader_tables(file_name)
    pheno_reader = phr.pheno_reader_tables(file_name)
    dataset = data.QTLData(geno_reader=geno_reader,pheno_reader=pheno_reader)

    K = dataset.getCovariance()
    if environments is None:
        phenotypes, sample_idx = dataset.getPhenotypes(center=False) 
    else:
        phenotypes, sample_idx = dataset.getPhenotypes(phenotype_IDs=environments, center=False) 

    snps = dataset.getGenotypes()
    Y = phenotypes.values

    individuals = np.array(list(phenotypes.index))

    environments = phenotypes.columns.values
    return Y, snps, K, individuals, dataset, environments



def get_data_with_parents(file_name, environments=None, N=None):

    geno_reader  = gr.genotype_reader_tables(file_name)
    pheno_reader = phr.pheno_reader_tables(file_name)
    dataset = data.QTLData(geno_reader=geno_reader,pheno_reader=pheno_reader)

    if N is None:
       subdataset = dataset
    else:
       Ntotal = dataset.sample_idx.shape[0]
       selected_indexes = sp.concatenate((sp.ones(N, dtype=bool), sp.zeros(Ntotal-N, dtype=bool)))
       sp.random.shuffle(selected_indexes)
       subdataset = dataset.subsample(rows = selected_indexes)

    K = subdataset.getCovariance()
    if environments is None:
        phenotypes, sample_idx = subdataset.getPhenotypes(center=False) 
    else:
        phenotypes, sample_idx = subdataset.getPhenotypes(phenotype_IDs=environments, center=False) 

    snps = subdataset.getGenotypes()
    Y = phenotypes.values

    individuals = np.array(list(phenotypes.index))
    parent1 = []
    parent2 = []
    for s in individuals:
        splitted = s.split("-")
        parent1.append(splitted[0])
        parent2.append(splitted[1])
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    if environments is None:
        environments = phenotypes.columns.values
        return Y, snps, K, parent1, parent2, individuals, subdataset, environments
    else:
        return Y, snps, K, parent1, parent2, subdataset
