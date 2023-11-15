#!/usr/bin/env python3
"""
Created on 2023-10-29 (Sun) 18:14:27

Sample codes for conducting PAM.

@author: I.Azuma
"""
# %%
import time
import pandas as pd
import seaborn as sns

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

import sys
sys.path.append('C:/github/PAM_Cython')
from pam_cython import pam
from pam_cython import pam_python_only

import lda
import lda.datasets

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
X.shape
X_df = pd.DataFrame(X,index=titles,columns=vocab)

# %% PAM with Cython
start = time.time()
S=5
K=10
dat = pam.PAM(S=S,K=K,alpha0=0.01,alpha1=0.01,beta=0.1,random_state=123)
dat.freq_df2bw(freq_df=X_df)
dat.set_params(seed_topics={},initial_conf=1.0)
dat.inference()

upper_doc_topic = pd.DataFrame(dat.get_theta0(),index=X_df.index,columns=['Topic {}'.format(i) for i in range(S)])
lower_doc_topic = pd.DataFrame(dat.get_theta1(),index=X_df.index,columns=['Topic {}'.format(i) for i in range(K)])
display(upper_doc_topic)
display(lower_doc_topic)

lower_topic_word = pd.DataFrame(dat.get_phi(),index=['Topic {}'.format(i) for i in range(K)],columns = X_df.columns)
end = time.time()
diff = end-start
print('Time: {} sec'.format(round(diff,4)))
# %% PAM with Python Only
start = time.time()
S=5
K=10
dat = pam_python_only.PAM(S=S,K=K,alpha0=0.01,alpha1=0.01,beta=0.1,random_state=123)
dat.freq_df2bw(freq_df=X_df)
dat.set_params(seed_topics={},initial_conf=1.0)
dat.inference()

upper_doc_topic = pd.DataFrame(dat.get_theta0(),index=X_df.index,columns=['Topic {}'.format(i) for i in range(S)])
lower_doc_topic = pd.DataFrame(dat.get_theta1(),index=X_df.index,columns=['Topic {}'.format(i) for i in range(K)])
display(upper_doc_topic)
display(lower_doc_topic)

lower_topic_word = pd.DataFrame(dat.get_phi(),index=['Topic {}'.format(i) for i in range(K)],columns = X_df.columns)
end = time.time()
diff = end-start
print('Time: {} sec'.format(round(diff,4)))
# %%
