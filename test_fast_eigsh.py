import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
from datasets import load_amazon_vg

A, users = load_amazon_vg()
d = np.array(A.sum(axis=1)).flatten()
D = sp.diags(d)
L = D - A

print("Starting eigsh for largest 15 eigenvalues...")
t0 = time.time()
vals, vecs = eigsh(L.astype(float), k=15, which='LA')
t1 = time.time()
print(f"Time taken: {t1 - t0:.2f} seconds")
print("Eigenvalues:", vals)
