#!/usr/bin/env python3
"""
Created on 2023-10-29 (Sun) 18:14:27

PAM dev

@author: I.Azuma
"""
# %%
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

import sys
sys.path.append('C:/github/PAM_Cython')
from pam_dev import pam

# %% PAM
X, _ = make_multilabel_classification(random_state=0,n_samples=200,n_features=100,n_classes=10)

dat = pam_deconv.PAMDeconv(S=2,K=3,alpha0=0.01,alpha1=0.01,beta=0.1,random_state=123)
dat.freq_df2bw(freq_df=X)
dat.set_params(seed_topics=seed_topics,initial_conf=1.0)

# %% ordinal LDA
X, _ = make_multilabel_classification(random_state=0,n_samples=200,n_features=100,n_classes=10)
lda = LatentDirichletAllocation(n_components=5,random_state=0)
lda.fit(X)
lda.transform(X[-2:])
