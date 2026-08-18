import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from datasets import load_ml1m
from scipy.stats import wasserstein_distance
from scipy.sparse.linalg import svds
import scipy.linalg as la

A, users = load_ml1m()
n = A.shape[0]
u, v = A.nonzero()
edges = [(r, c) for r, c in zip(u, v) if r < c]
np.random.shuffle(edges)
test_size = 2000
test_pos = edges[:test_size]
train_edges = edges[test_size:]

train_rows, train_cols = [], []
for r, c in train_edges:
    train_rows.extend([r, c])
    train_cols.extend([c, r])
A_train = sp.csr_matrix(([1]*len(train_rows), (train_rows, train_cols)), shape=(n, n))

test_neg = []
while len(test_neg) < test_size:
    r = np.random.randint(0, users)
    c = np.random.randint(users, n)
    if A[r, c] == 0:
        test_neg.append((r, c))

d = np.array(A_train.sum(axis=1)).flatten()
D = sp.diags(d)
L = (D - A_train).toarray()
vals, vecs = la.eigh(L, subset_by_index=[1, 16])

F_norm = vecs # NO NORMALIZATION!

alpha = 0.5
node_supports = []
for u_node in range(n):
    neighbors = A_train.indices[A_train.indptr[u_node]:A_train.indptr[u_node+1]]
    deg_u = len(neighbors)
    if deg_u > 0:
        supp = [u_node] + list(neighbors)
        w = [alpha] + [(1-alpha)/deg_u]*deg_u
    else:
        supp = [u_node]; w = [1.0]
    node_supports.append((np.array(supp), np.array(w)))
    
def score_mean(u, v):
    su, wu = node_supports[u]
    sv, wv = node_supports[v]
    return 1.0 - np.mean([wasserstein_distance(F_norm[su, i], F_norm[sv, i], wu, wv) for i in range(F_norm.shape[1])])

pos_scores = [score_mean(u, v) for u, v in test_pos]
neg_scores = [score_mean(u, v) for u, v in test_neg]
y_true = [1]*test_size + [0]*test_size
y_scores = np.array(pos_scores + neg_scores)
auc = roc_auc_score(y_true, y_scores)
if auc < 0.5: auc = roc_auc_score(y_true, -y_scores)
print(f"No Norm Mean AUC: {auc}")
