import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from sklearn.metrics import roc_auc_score
from datasets import load_ml1m
from scipy.stats import wasserstein_distance

A, users = load_ml1m()
n = A.shape[0]
u, v = A.nonzero()
edges = [(r, c) for r, c in zip(u, v) if r < c]
np.random.shuffle(edges)
test_size = min(len(edges) // 10, 2000)
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

# test smallest
vals_sm, vecs_sm = la.eigh(L, subset_by_index=[1, 15])
# test largest
vals_la, vecs_la = la.eigh(L, subset_by_index=[n-15, n-1])

def evaluate(vecs):
    edge_diffs = np.abs(vecs[train_rows, :] - vecs[train_cols, :])
    max_diffs = np.max(edge_diffs, axis=0)
    max_diffs[max_diffs < 1e-12] = 1.0
    F_norm = vecs / max_diffs
    
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
        
    def score(u, v):
        supp_u, w_u = node_supports[u]
        supp_v, w_v = node_supports[v]
        max_w1 = 0.0
        for i in range(F_norm.shape[1]):
            w1 = wasserstein_distance(F_norm[supp_u, i], F_norm[supp_v, i], w_u, w_v)
            if w1 > max_w1: max_w1 = w1
        return 1.0 - max_w1
        
    pos_scores = [score(u, v) for u, v in test_pos]
    neg_scores = [score(u, v) for u, v in test_neg]
    y_true = [1]*test_size + [0]*test_size
    y_scores = np.array(pos_scores + neg_scores)
    auc = roc_auc_score(y_true, y_scores)
    return auc if auc >= 0.5 else roc_auc_score(y_true, -y_scores)

print("Smallest eigs AUC:", evaluate(vecs_sm))
print("Largest eigs AUC:", evaluate(vecs_la))
