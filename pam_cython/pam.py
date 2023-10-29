# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:29:21 2022

Pachinko Allocation Model

@author: I.Azuma
"""
import logging
logger = logging.getLogger('pam')

import random
import itertools
import numpy as np
from scipy import special

from pam_cython import vocabulary
from pam_cython import utils
from pam_cython import _pam

from pathlib import Path
BASE_DIR = Path(__file__).parent
print(BASE_DIR)
print("!! Pachinko Allocation Model!!")

class PAM():
    """
    Pachinko Allocation Model using Gibbs sampling
    """
    
    def __init__(self, bw=None, S=1, K=3, n_iter=2000, alpha0=1.0, alpha1=1.0, beta=1.0, random_state=None, refresh=10):
        self.bw = bw
        self.S = S
        self.K = K
        self.n_iter = n_iter
        # hyper parameter for upper topic (1, S)
        self.alpha0 = np.zeros(self.S) + alpha0
        # hyper parameter for lower topic (S, K)
        self.alpha1 = np.zeros(shape=(S, K)) + alpha1
        self.beta = beta
        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh
        
        # word distribution
        self.phi = None
        # upper topic distribution
        self.theta0 = None
        # lower topic distribution
        self.theta1 = None

        if alpha0 <= 0 or alpha1 <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be greater than zero")

        # random numbers that are reused
        rng = utils.check_random_state(random_state)
        if random_state:
            random.seed(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates

        # configure console logging if not already configured
        if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
            logging.basicConfig(level=logging.INFO)
    
    def freq_df2bw(self,freq_df):
        """
        freq_df : DataFrame
            samples in row and genes in column. Gene expression level is contained in int type
        
                    word1  word2  ...  word49  word50
        sample: 1    8      7     ...     4     4
        sample: 2    8      7     ...     4     4
        sample: 3    8      7     ...     4     4
        sample: 4    6      5     ...     3     3
        sample: 5    6      5     ...     3     3
        """
        freq_df = freq_df.astype(int)
        
        # seed-topic preparation
        gene_names = [t.upper() for t in freq_df.columns.tolist()]
        samples = freq_df.index.tolist()
        gene2id = dict((v, idx) for idx, v in enumerate(gene_names))

        # prepare docs
        docs = []
        for idx in range(len(freq_df)):
            vocab = freq_df.columns.tolist()
            freq = freq_df.iloc[idx,0::]
            tmp_doc = [[vocab[i]]*freq[i] for i in range(len(vocab))]
            doc = list(itertools.chain.from_iterable(tmp_doc))
            doc_id = [gene2id.get(t) for t in doc]
            docs.append(doc_id)
        
        #voca = vocabulary.Vocabulary()
        #use_docs = [voca.doc_to_ids(doc) for doc in docs]
        # padding
        l = max(map(len,docs))
        self.bw = np.array([x+[-1]*(l-len(x)) for x in docs],dtype=np.intc)
        #self.bw = np.array(use_docs)
    
    def freq_df2bw_legacy(self,freq_df):
        """
        freq_df : DataFrame
            samples in row and genes in column. Gene expression level is contained in int type
        
                    MBP  PTGDS  ...  PPIC  LOX
        tissue: 1    8      7  ...     4    4
        tissue: 2    8      7  ...     4    4
        tissue: 3    8      7  ...     4    4
        tissue: 4    6      5  ...     3    3
        tissue: 5    6      5  ...     3    3
        """
        freq_df = freq_df.astype(int)
        # prepare docs
        docs = []
        for idx in range(len(freq_df)):
            vocab = freq_df.columns.tolist()
            freq = freq_df.iloc[idx,0::]
            tmp_doc = [[vocab[i]]*freq[i] for i in range(len(vocab))]
            doc = list(itertools.chain.from_iterable(tmp_doc))
            docs.append(doc)
        
        voca = vocabulary.Vocabulary()
        use_docs = [voca.doc_to_ids(doc) for doc in docs]
        # padding
        l = max(map(len,use_docs))
        self.bw = np.array([x+[-1]*(l-len(x)) for x in use_docs],dtype=np.intc)
        #self.bw = np.array(use_docs)
    
    # D_S_K aggregation
    def set_params(self,seed_topics={},initial_conf=1.0):
        # vocabulary number
        self.V = np.max([word for words in self.bw for word in words]) + 1
        # document number
        self.D = len(self.bw)
        # initialize Topics assigned to each topic word, (D, W, 2), randomly
        np.random.seed(self.random_state)
        self.z_s_k = np.array([[[np.random.randint(0, self.S), np.random.randint(0, self.K)]
                       for w in words] for words in self.bw],dtype=np.intc)
        # Aggregation of words assigned to S and K topics for each document
        self.D_S_K = np.zeros(shape=(self.D, self.S, self.K),dtype=np.intc)
        # Number of words in lower topic
        self.K_V = np.zeros(shape=(self.K, self.V),dtype=np.intc)
        
        max_v = max(list(seed_topics.values()))
        if self.K < (max_v + 1):
            raise ValueError("!! Lack of lower topics !!")
        
        # initialize
        """ without guide
        for d, words in enumerate(self.bw):
            for w, word in enumerate(words):
                if word < 0:
                    pass
                else:
                    self.D_S_K[d][self.z_s_k[d][w][0]][self.z_s_k[d][w][1]] += 1
                    self.K_V[self.z_s_k[d][w][1]][word] += 1
        """
        # initialization
        for d, words in enumerate(self.bw):
            for w, word in enumerate(words):
                if word < 0:
                    pass
                else:
                    # seeded
                    if word in seed_topics and random.random() < initial_conf: # # 0 <= random.random() < 1
                        z_new = seed_topics[word] # set the seed topic to lower layer
                        self.z_s_k[d][w][1] = z_new # change the randomized topic to the seed one
                    # non-seeded
                    else:
                        z_new = self.z_s_k[d][w][1]
                    self.D_S_K[d][self.z_s_k[d][w][0]][z_new] += 1
                    self.K_V[z_new][word] += 1
    
    def infer_z(self):
        """
        Cython
        """
        alpha0 = self.alpha0
        alpha1 = self.alpha1
        beta = self.beta
        probs = np.zeros(self.S*self.K)
        _pam._infer_z(self.bw, self.D_S_K, self.K_V, self.z_s_k, probs, alpha0, alpha1, beta)
        #_pam._infer_z_legacy(self.bw, self.D_S_K, self.K_V, self.z_s_k, probs, alpha0, alpha1, beta)
        #_pam._infer_z_python(self.bw, self.D_S_K, self.K_V, self.z_s_k, probs, alpha0, alpha1, beta)
    
    # estimate alpha0 (MLP version)
    def infer_alpha0(self):
        # update alpha0
        alpha0_base = np.zeros(self.S)
        for s in range(self.S):
            mole = np.sum([special.digamma(np.sum(self.D_S_K[d][s])+self.alpha0[s])
                           for d in range(self.D)]) \
                 - self.D * special.digamma(self.alpha0[s])
            deno = np.sum([special.digamma(np.sum(self.D_S_K[d])+np.sum(self.alpha0))
                           for d in range(self.D)]) \
                 - self.D * special.digamma(np.sum(self.alpha0))
            #if mole != 0:
                #alpha0_base[s] = mole / deno
            if mole > 0:
                alpha0_base[s] = mole / deno
            else:
                alpha0_base[s] = 1.0
        self.alpha0 *= alpha0_base

    # estimate alpha1 (MLP version)
    def infer_alpha1(self):
        # update alpha1
        alpha1_base = np.zeros(shape=(self.S, self.K))
        for s in range(self.S):
            for k in range(self.K):
                mole = np.sum([special.digamma(self.D_S_K[d][s][k]+self.alpha1[s][k])
                               for d in range(self.D)]) \
                     - self.D * special.digamma(self.alpha1[s][k])
                deno = np.sum([special.digamma(np.sum(self.D_S_K[d][s])+np.sum(self.alpha1[s]))
                               for d in range(self.D)]) \
                     - self.D * special.digamma(np.sum(self.alpha1[s]))
                # avoid alpha is equal to 0
                #if mole != 0:
                    #alpha1_base[s][k] = mole / deno
                if mole > 0:
                    alpha1_base[s][k] = mole / deno
                else:
                    alpha1_base[s][k] = 1.0
        self.alpha1 *= alpha1_base
    
    # estimate beta
    def infer_beta(self):
        mole = np.sum(special.digamma(self.K_V+self.beta)) \
             - self.K * self.V * special.digamma(self.beta)
        deno = self.V * np.sum(special.digamma(np.sum(self.K_V, axis=1)+self.beta*self.V)) \
             - self.K * self.V * special.digamma(self.beta*self.V)
        self.beta *= mole / deno
    
    # calculation part
    def inference(self):
        # word's topic sampling
        self.infer_z()
        # update alpha
        self.infer_alpha0()
        self.infer_alpha1()
        # update beta
        self.infer_beta()
    
    # calculate word distribution
    def cal_phi(self):
        self.phi = (self.K_V + self.beta) / (np.sum(self.K_V, axis=1)[:,np.newaxis] + self.beta * self.V)

    # obtain word dist
    def get_phi(self):
        if self.phi == None:
            self.cal_phi()
        return self.phi

    # calculate upper topic distribution
    def cal_theta0(self):
        mole = np.array([np.sum(S_K, axis=1)+self.alpha0 for S_K in self.D_S_K]) # lower topic wide
        deno = np.sum(mole, axis=1)[:, np.newaxis]
        self.theta0 = mole / deno

    # obtain upper topic dist
    def get_theta0(self):
        if self.theta0 == None:
            self.cal_theta0()
        return self.theta0
    
    # calculate lower topic distribution
    def cal_theta1(self):
        tmp_mole = np.array([np.sum(S_K, axis=0) for S_K in self.D_S_K]) # upper topic wide
        lower_alpha = np.sum(self.alpha1,axis=0)
        mole = np.array([t+lower_alpha for t in tmp_mole])
        deno = np.sum(mole, axis=1)[:, np.newaxis]
        self.theta1 = mole / deno
    
    # obtain lower topic dist
    def get_theta1(self):
        if self.theta1 == None:
            self.cal_theta1()
        return self.theta1

    # obtain alpha1
    def get_alpha1(self):
        return self.alpha1