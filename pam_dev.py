#!/usr/bin/env python3
"""
Created on 2023-10-29 (Sun) 18:14:27

PAM dev

@author: I.Azuma
"""
# %%
import pandas as pd
import seaborn as sns

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

import sys
sys.path.append('C:/github/PAM_Cython')
from pam_cython import pam

# %% PAM
n_samples = 100
n_features = 50
X, _ = make_multilabel_classification(random_state=0,n_samples=n_samples,n_features=n_features, n_classes=10)
X_df = pd.DataFrame(X,index=['sample_{}'.format(i) for i in range(n_samples)],columns=['word_{}'.format(i) for i in range(n_features)])
sns.clustermap(X_df)

dat = pam.PAM(S=2,K=5,alpha0=0.01,alpha1=0.01,beta=0.1,random_state=123)
dat.freq_df2bw(freq_df=X_df)
dat.set_params(seed_topics={},initial_conf=1.0)
dat.inference()

res0 = pd.DataFrame(dat.get_theta0(),index=X_df.index)
res1 = pd.DataFrame(dat.get_theta1(),index=X_df.index)

# %%
import lda
import lda.datasets

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
X.shape
X_df = pd.DataFrame(X,index=titles,columns=vocab)

# %%
S=2
K=5
dat = pam.PAM(S=S,K=K,alpha0=0.01,alpha1=0.01,beta=0.1,random_state=123)
dat.freq_df2bw(freq_df=X_df)
dat.set_params(seed_topics={},initial_conf=1.0)
dat.inference()

upper_doc_topic = pd.DataFrame(dat.get_theta0(),index=X_df.index,columns=['Topic {}'.format(i) for i in range(S)])
lower_doc_topic = pd.DataFrame(dat.get_theta1(),index=X_df.index,columns=['Topic {}'.format(i) for i in range(K)])
display(upper_doc_topic)
display(lower_doc_topic)

lower_topic_word = pd.DataFrame(dat.get_phi(),index=['Topic {}'.format(i) for i in range(K)],columns = X_df.columns)

# %%
