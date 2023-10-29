#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
import numpy as np

cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)


cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin
        
cdef int calc_int_sum(int[:] target, int length) nogil:
    cdef int i
    cdef int target_sum = 0
    
    for i in range(length):
        target_sum += target[i]
    return target_sum
    
cdef double calc_double_sum(double[:] target, int length) nogil:
    cdef int i
    cdef double target_sum = 0
    
    for i in range(length):
        target_sum += target[i]
    return target_sum
        
def _infer_z(int[:, :] bw, int[:, :, :] dsk, int[:, :] kv, int[:, :, :] zsk, double[:] probs,
             double[:] alpha0, double[:,:] alpha1, double beta):
    cdef int i, j, w, u, l, p, s, k, v, Nds, Nk, Ndsk, Nkw, s_k
    cdef int D = bw.shape[0]
    cdef int W = bw.shape[1]
    cdef int V = kv.shape[1]
    cdef int S = dsk.shape[1]
    cdef int K = dsk.shape[2]
    cdef double prob, prob_sum, alpha_sum
    
    with nogil:
        for i in range(D):
            for j in range(W):
                w = bw[i][j] # value of jth word in ith document
                u = zsk[i][j][0] # upper topic idx
                l = zsk[i][j][1] # lower topic idx
                if w < 0:
                    pass
                else:
                    dec(dsk[i, u, l])
                    dec(kv[l, w])
                    # initialize
                    for p in range(S*K):
                        probs[p] = 0
                    # calculate p(z=s,k)
                    prob_sum = 0
                    for s in range(S):
                        for k in range(K):
                            Nds = calc_int_sum(dsk[i][s],K)
                            Nk = calc_int_sum(kv[k],kv.shape[1])
                            Ndsk = dsk[i][s][k]
                            Nkw = kv[k][w]
                            alpha_sum = calc_double_sum(alpha1[s],K)
                            prob = ((Nds + alpha0[s]) * (Ndsk + alpha1[s][k]) * (Nkw + beta)) / ((Nk + beta * V) * (Nds + alpha_sum))

                            probs[K*s + k] = prob # update probs
                            prob_sum += prob
                    
                    for p in range(S*K):
                        probs[p] /= prob_sum # normalize
                    # sampling
                    with gil:
                        try:
                            s_k = np.random.multinomial(1, probs).argmax()
                        except:
                            print(probs)
                            pass
                    zsk[i][j][0] = s_k / K
                    zsk[i][j][1] = s_k % K
                    inc(dsk[i, zsk[i][j][0], zsk[i][j][1]])
                    inc(kv[zsk[i][j][1], w])

def _infer_z_legacy(int[:, :] bw, int[:, :, :] dsk, int[:, :] kv, int[:, :, :] zsk, double[:] probs,
             double[:] alpha0, double[:,:] alpha1, double beta):
    cdef int i, j, w, u, l, s, k, Nds, Nk, Ndsk, Nkw, s_k
    cdef int D = bw.shape[0]
    cdef int V = kv.shape[1]
    cdef int S = dsk.shape[1]
    cdef int K = dsk.shape[2]
    cdef double prob
    cdef double prob_sum
    
    with nogil:
        for i in range(D):
            for j in range(V):
                w = bw[i][j] # value of jth word in ith document
                u = zsk[i][j][0] # upper topic idx
                l = zsk[i][j][1] # lower topic idx
                if w <0:
                    pass
                else:
                    dec(dsk[i, u, l])
                    dec(kv[l, w])
                    # initialize
                    for p in range(S*K):
                        probs[p] = 0
                    # calculate p(z=s,k)
                    prob_sum = 0
                    for s in range(S):
                        for k in range(K):
                            prob = 0
                            with gil:
                                Nds = np.sum(dsk[i][s])
                                Nk = np.sum(kv[k])
                                Ndsk = dsk[i][s][k]
                                Nkw = kv[k][w]
                                prob += (Nds + alpha0[s]) * ((Ndsk + alpha1[s][k]) / (Nds + np.sum(alpha1[s]))) * ((Nkw + beta) / (Nk + beta * V))
                            probs[K*s + k] = prob # update probs
                            prob_sum += prob
                    
                    for p in range(S*K):
                        probs[p] /= prob_sum # normalize
                    # sampling
                    with gil:
                        s_k = np.random.multinomial(1, probs).argmax()
                    zsk[i][j][0] = s_k/K
                    zsk[i][j][1] = s_k%K
                    inc(dsk[i, zsk[i][j][0], zsk[i][j][1]])
                    inc(kv[zsk[i][j][1], w])

def _infer_z_python(int[:, :] bw, int[:, :, :] dsk, int[:, :] kv, int[:, :, :] zsk, double[:] probs,
             double[:] alpha0, double[:,:] alpha1, double beta):
    cdef int S = dsk.shape[1]
    cdef int K = dsk.shape[2]
    cdef int V = kv.shape[1]
    
    for d, words in enumerate(bw):
        for w, word in enumerate(words):
            if word < 0:
                pass
            else:
                # remove the target word
                dsk[d][zsk[d][w][0]][zsk[d][w][1]] -= 1
                kv[zsk[d][w][1]][word] -= 1
                # calculate p(z = s, k)
                probs = np.zeros(S*K)
                for s in range(S):
                    for k in range(K):
                        N_ds = np.sum(dsk[d][s])
                        N_dsk = dsk[d][s][k]
                        N_k = np.sum(kv[k])
                        N_kw = kv[k][word]
                        #print N_dsk + self.alpha1[s][k]
                        #print N_ds + np.sum(self.alpha1[s])
                        prob = (N_ds + alpha0[s]) \
                             * ((N_dsk + alpha1[s][k]) / (N_ds + np.sum(alpha1[s]))) \
                             * ((N_kw + beta) / (N_k + beta * V))
                        #if prob < 0:
                            #prob = np.spacing(1)
                        probs[K*s + k] = prob
                probs /= np.sum(probs)
                # sampling
                try:
                    s_k = np.random.multinomial(1, probs).argmax()
                    # update count
                    zsk[d][w][0] = s_k // K
                    zsk[d][w][1] = s_k % K
                    dsk[d][zsk[d][w][0]][zsk[d][w][1]] += 1
                    kv[zsk[d][w][1]][word] += 1
                except:
                    print(probs)


cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double eta) nogil:
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta, lgamma_alpha
    with nogil:
        lgamma_eta = lgamma(eta)
        lgamma_alpha = lgamma(alpha)

        ll += n_topics * lgamma(eta * vocab_size)
        for k in range(n_topics):
            ll -= lgamma(eta * vocab_size + nz[k])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += lgamma(eta + nzw[k, w]) - lgamma_eta

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += lgamma(alpha + ndz[d, k]) - lgamma_alpha
        return ll
