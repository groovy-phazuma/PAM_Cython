# PAM_Cython
Pachinko Allocation Model (PAM)
- Upper topic distribution $θ_1$ for S topics.
- Lower topic distribution $θ_2$ for K topics.
- Upper topic $z_1$ and lower topic $z_2$. Infer z using Cython.

## Setup
Build for Cython file.
```
pip install cython
cd ./PAM_Cython/pam_cython
python setup.py build_ext --inplace
```
※ If an error ```'gcc' failed: No such file or directory``` appears, perform  ```sudo apt-get install gcc```.

## Getting started
```
>>> import numpy as np
>>> import lda
>>> import lda.datasets
>>> X = lda.datasets.load_reuters()
>>> vocab = lda.datasets.load_reuters_vocab()
>>> titles = lda.datasets.load_reuters_titles()
>>> X.shape
(395, 4258)
>>> X.sum()
84010

>>> S=2
>>> K=5
>>> model = pam.PAM(S=S,K=K,alpha0=0.01,alpha1=0.01,beta=0.1,random_state=123)
>>> model.freq_df2bw(freq_df=X_df)
>>> model.set_params(seed_topics={},initial_conf=1.0)
>>> model.inference()
```
The document-topic distributions are available in `model.theta0` and `model.theta1`.
```
                                                                                          Topic 0	Topic 1         Topic 2	        Topic 3	        Topic 4
0 UK: Prince Charles spearheads British royal revolution. LONDON 1996-08-20	          0.197374	0.298069	0.162348	0.149214	0.192995
1 GERMANY: Historic Dresden church rising from WW2 ashes. DRESDEN, Germany 1996-08-21	  0.161880	0.139886	0.264511	0.205865	0.227858
2 INDIA: Mother Teresa's condition said still unstable. CALCUTTA 1996-08-23	          0.198316	0.202527	0.282558	0.168830	0.147770
3 UK: Palace warns British weekly over Charles pictures. LONDON 1996-08-25	          0.251108	0.145035	0.226999	0.169143	0.207715
4 INDIA: Mother Teresa, slightly stronger, blesses nuns. CALCUTTA 1996-08-25	          0.152489	0.156200	0.156200	0.223013	0.312098
```
The topic-word distributions are available in `model.phi`.
```
          church	pope	        years	        people	        mother	        last	   ...
Topic 0	  0.006362	0.005612	0.004573	0.002898	0.004111	0.003476   ...
Topic 1	  0.007341	0.003534	0.003932	0.003250	0.003307	0.004614   ...
Topic 2	  0.006742	0.008021	0.004739	0.003015	0.004016	0.002681   ...
Topic 3	  0.005730	0.007356	0.002590	0.004272	0.004160	0.003319   ...
Topic 4	  0.009583	0.005709	0.005025	0.005880	0.003031	0.003829   ...
```

## Dependency
- Python = 3.8
- Cython = 3.0.5
- requirements: pandas, numpy, lda (for toy data)

## References
- Lin and MacCallum, 'Pachinko Allocation: DAG-Structured Mixture Models of Topic Correlations', ICML, 2006.
- https://github.com/kenchin110100/machine_learning
