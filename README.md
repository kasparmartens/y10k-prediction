# Predicting quantitative traits from genome and phenome with near perfect accuracy

Python code and IPython notebooks accompanying [our paper](http://biorxiv.org/content/early/2015/10/26/029868). 

Specifically, 

- `train_and_test_sets.py` - code for partitioning individuals into four sets, as shown in Figure 3a
- `BLUP.py` - fitting the BLUP model
- `QTL_fitting.py` - constructing and fitting the QTL model
- `LMM.py` - constructing and fitting the LMM and LMM+P models
- `MTLMM.py` - fitting the multi-trait LMM
- `MRF.py` - fitting the mixed random forest 

[This notebook](results2ab.ipynb) was used to produce the results for Figure 2. 
All models were fitted using [Limix](https://github.com/PMBio/limix). 
