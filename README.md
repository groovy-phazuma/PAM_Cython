# PAM_Cython
Pachinko Allocation Model (PAM)
- Upper topic distribution $θ_1$ for S topics.
- Lower topic distribution $θ_2$ for K topics.
- Upper topic $z_1$ and lower topic $z_2$. Infer z using Cython.

## Usage
Build for Cython file.
```
pip install cython
cd ./PAM_Cython/pam_cython
python setup.py build_ext --inplace
```
※ If an error ```'gcc' failed: No such file or directory``` appears, perform  ```sudo apt-get install gcc```.

## References
- Lin and MacCallum, 'Pachinko Allocation: DAG-Structured Mixture Models of Topic Correlations', ICML, 2006.
- https://github.com/kenchin110100/machine_learning
