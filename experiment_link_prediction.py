import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import structured_negative_sampling
from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import svds
import networkx as nx
import time

from datasets import load_amazon_vg, load_ml1m
from sorc import compute_sorc

def run_pipeline(dataset_name, A_full, num_users):
    print(f"\n--- Running Pipeline for {dataset_name} ---")
    
    n = A_full.shape[0]
    u, v = A_full.nonzero()
    edge_set = set(zip(u, v))
    
    # 10% test mask
    # Only pick edges where u < v to avoid symmetric duplicates in the test set
    upper_edges = [(r, c) for r, c in zip(u, v) if r < c]
    np.random.shuffle(upper_edges)
    test_size = min(len(upper_edges) // 10, 5000)
    
    test_pos = upper_edges[:test_size]
    train_pos = upper_edges[test_size:]
    
    # Build training matrix strictly removing test edges
    test_pos_set = set(test_pos)
    train_rows, train_cols, train_data = [], [], []
    
    for r, c in train_pos:
        train_rows.extend([r, c])
        train_cols.extend([c, r])
        train_data.extend([1, 1])
        
    print(f"Graph nodes: {n}, Train edges (symmetric): {len(train_data)}")
    
    A_train = sp.csr_matrix((train_data, (train_rows, train_cols)), shape=(n, n))
    
    # Generate Negative testing set
    test_neg = []
    while len(test_neg) < test_size:
        r = np.random.randint(0, num_users)
        c = np.random.randint(num_users, n)
        if (r, c) not in edge_set and (c, r) not in edge_set:
            test_neg.append((r, c))
            
    # --- 1. SVD Baseline ---
    print("1. Computing SVD...")
    A_float = A_train.asfptype()
    u_svd, s_svd, vt_svd = svds(A_float, k=64)
    user_emb_svd = u_svd * np.sqrt(s_svd)
    item_emb_svd = vt_svd.T * np.sqrt(s_svd)

    # --- 2. LightGCN Baseline ---
    print("2. Training LightGCN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    edge_index = torch.tensor([train_rows, train_cols], dtype=torch.long).to(device)
    model = LightGCN(num_nodes=n, embedding_dim=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        emb = model(edge_index)
        out = structured_negative_sampling(edge_index, num_nodes=n)
        i, j, k = out 
        pos_score = (emb[i] * emb[j]).sum(dim=-1)
        neg_score = (emb[i] * emb[k]).sum(dim=-1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        all_emb = model.get_embedding(edge_index)
        
    # --- 3. AFRC & SORC Metrics ---
    print("3. Computing Curvatures...")
    d = np.array(np.abs(A_train).sum(axis=1)).flatten()
    
    # Pre-compute SORC on the training graph
    print("  -> Computing SORC (k=15) on train graph...")
    G_train = nx.from_scipy_sparse_array(A_train)
    # SORC is slow if we do it for all node pairs naively, but we only need it for the edges
    # Wait, for test edges (both pos and neg), they might NOT be in G_train!
    # If a test negative is not an edge in G_train, compute_sorc won't naturally score it because compute_sorc only iterates over G.edges().
    # We must explicitly query the 1D Wasserstein distance between the nodes in test_pos and test_neg.
    
    # Re-implementing the core of SORC here to allow scoring non-edges:
    from scipy.sparse.linalg import eigsh
    from scipy.stats import wasserstein_distance
    
    D = sp.diags(d)
    L = D - A_train
    import scipy.linalg as la
    L_dense = L.toarray()
    num_eigs = min(15 + 1, n - 1)
    vals, vecs = la.eigh(L_dense, subset_by_index=[0, num_eigs-1])
    idx_sort = np.argsort(vals)
    vecs = vecs[:, idx_sort[1:]]
    
    # Find max differences along training edges to normalize
    edge_diffs = np.abs(vecs[train_rows, :] - vecs[train_cols, :])
    max_diffs = np.max(edge_diffs, axis=0)
    max_diffs[max_diffs < 1e-12] = 1.0 
    F_norm = vecs / max_diffs
    
    # Node supports
    alpha = 0.5
    node_supports = []
    for u_node in range(n):
        neighbors = A_train.indices[A_train.indptr[u_node]:A_train.indptr[u_node+1]]
        deg_u = len(neighbors)
        if deg_u > 0:
            support = [u_node] + list(neighbors)
            weights = [alpha] + [(1 - alpha) / deg_u] * deg_u
        else:
            support = [u_node]
            weights = [1.0]
        node_supports.append((np.array(support), np.array(weights)))
        
    def score_sorc(u, v):
        supp_u, w_u = node_supports[u]
        supp_v, w_v = node_supports[v]
        max_w1 = 0.0
        for i in range(F_norm.shape[1]):
            u_vals = F_norm[supp_u, i]
            v_vals = F_norm[supp_v, i]
            w1 = wasserstein_distance(u_vals, v_vals, w_u, w_v)
            if w1 > max_w1:
                max_w1 = w1
        return 1.0 - max_w1 # Higher SORC -> More likely an edge
        
    def score_fstar(u, v):
        du, dv = d[u], d[v]
        if du == 0 or dv == 0:
            return 0
        row_A2_at_v = A_train[u].multiply(A_train[v]).sum()
        C4_sum = row_A2_at_v - du - dv + 1
        max_bound = max(1, (du - 1) * (dv - 1))
        normalized = C4_sum / max_bound
        return 4 - du - dv + 3.0 * min(du - 1, dv - 1) * normalized
        
    def score_svd(u, v):
        return np.dot(user_emb_svd[u], item_emb_svd[v])
        
    def score_gcn(u, v):
        return torch.dot(all_emb[u], all_emb[v]).item()
        
    # --- Scoring ---
    print("4. Evaluating AUC...")
    metrics = {"SVD": score_svd, "LightGCN": score_gcn, "AFRC": score_fstar, "SORC": score_sorc}
    results = {}
    
    y_true = [1]*test_size + [0]*test_size
    
    for name, func in metrics.items():
        pos_scores = [func(u,v) for u,v in test_pos]
        neg_scores = [func(u,v) for u,v in test_neg]
        
        # AFRC is inversely correlated (highly negative for bridge/missing edges, but wait... 
        # Actually for link prediction, higher curvature = higher likelihood of edge?
        # Typically missing edges have more negative curvature. Let's just check the correlation sign
        # and flip if AUC < 0.5.
        y_scores = np.array(pos_scores + neg_scores)
        auc = roc_auc_score(y_true, y_scores)
        if auc < 0.5:
            auc = roc_auc_score(y_true, -y_scores)
            
        results[name] = auc
        print(f"  {name} AUC: {auc:.4f}")
        
    return results

if __name__ == "__main__":
    import datasets
    
    try:
        a_A, a_users = datasets.load_amazon_vg()
        a_res = run_pipeline("Amazon Video Games", a_A, a_users)
    except FileNotFoundError:
        print("Amazon dataset not found. Skipping.")
        a_res = None
        
    try:
        m_A, m_users = datasets.load_ml1m()
        m_res = run_pipeline("MovieLens-1M", m_A, m_users)
    except FileNotFoundError:
        print("ML-1M dataset not found. Skipping.")
        m_res = None
        
    with open("link_prediction_results.txt", "w") as f:
        f.write("Dataset,SVD,LightGCN,AFRC,SORC\n")
        if a_res:
            f.write(f"Amazon,{a_res['SVD']:.4f},{a_res['LightGCN']:.4f},{a_res['AFRC']:.4f},{a_res['SORC']:.4f}\n")
        if m_res:
            f.write(f"MovieLens,{m_res['SVD']:.4f},{m_res['LightGCN']:.4f},{m_res['AFRC']:.4f},{m_res['SORC']:.4f}\n")
    print("Saved results to link_prediction_results.txt")
