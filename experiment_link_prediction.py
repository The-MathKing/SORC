import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import structured_negative_sampling
from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import svds, eigsh
import networkx as nx
from scipy.stats import wasserstein_distance
import time

from datasets import load_amazon_vg, load_ml1m

def run_pipeline(dataset_name, A_full, num_users):
    print(f"\n--- Running Pipeline for {dataset_name} ---")
    
    n = A_full.shape[0]
    u, v = A_full.nonzero()
    edge_set = set(zip(u, v))
    upper_edges = [(r, c) for r, c in zip(u, v) if r < c]
    
    num_folds = 5
    auc_results = {'SVD': [], 'LightGCN': [], 'AFRC': [], 'SORC': []}
    
    for fold in range(num_folds):
        print(f"  -> Fold {fold+1}/{num_folds}...")
        np.random.shuffle(upper_edges)
        test_size = min(len(upper_edges) // 10, 5000)
        
        test_pos = upper_edges[:test_size]
        train_pos = upper_edges[test_size:]
        
        train_rows, train_cols, train_data = [], [], []
        for r, c in train_pos:
            train_rows.extend([r, c])
            train_cols.extend([c, r])
            train_data.extend([1, 1])
            
        A_train = sp.csr_matrix((train_data, (train_rows, train_cols)), shape=(n, n))
        
        test_neg = []
        while len(test_neg) < test_size:
            r = np.random.randint(0, num_users)
            c = np.random.randint(num_users, n)
            if (r, c) not in edge_set and (c, r) not in edge_set:
                test_neg.append((r, c))
                
        # 1. Computing SVD
        u_svd, s_svd, vt_svd = svds(A_train.asfptype(), k=64)
        user_emb_svd = u_svd * np.sqrt(s_svd)
        item_emb_svd = vt_svd.T * np.sqrt(s_svd)
        def score_svd(u, v): return np.dot(user_emb_svd[u], item_emb_svd[v])

        # 2. LightGCN
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
        def score_gcn(u, v): return torch.dot(all_emb[u], all_emb[v]).item()

        # 3. AFRC & SORC
        d = np.array(np.abs(A_train).sum(axis=1)).flatten()
        D = sp.diags(d)
        L = D - A_train
        
        # Using eigsh directly for massive speedup
        num_eigs = min(15, n - 1)
        vals, vecs = eigsh(L.astype(float), k=num_eigs, which='LA')
        
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
                support = [u_node] + list(neighbors)
                weights = [alpha] + [(1 - alpha) / deg_u] * deg_u
            else:
                support = [u_node]
                weights = [1.0]
            node_supports.append((np.array(support), np.array(weights)))
            
        def score_sorc(u, v):
            supp_u, w_u = node_supports[u]
            supp_v, w_v = node_supports[v]
            w1_sum = 0.0
            for i in range(F_norm.shape[1]):
                u_vals = F_norm[supp_u, i]
                v_vals = F_norm[supp_v, i]
                w1_sum += wasserstein_distance(u_vals, v_vals, w_u, w_v)
            mean_w1 = w1_sum / F_norm.shape[1]
            return 1.0 - mean_w1
            
        def score_fstar(u, v):
            du, dv = d[u], d[v]
            if du == 0 or dv == 0: return 0
            row_A2_at_v = A_train[u].multiply(A_train[v]).sum()
            C4_sum = row_A2_at_v - du - dv + 1
            max_bound = max(1, (du - 1) * (dv - 1))
            normalized = C4_sum / max_bound
            return 4 - du - dv + 3.0 * min(du - 1, dv - 1) * normalized
            
        y_true = [1]*test_size + [0]*test_size
        test_pairs = test_pos + test_neg
        metrics = {"SVD": score_svd, "LightGCN": score_gcn, "AFRC": score_fstar, "SORC": score_sorc}
        
        for name, func in metrics.items():
            scores = np.array([func(u,v) for u,v in test_pairs])
            auc = roc_auc_score(y_true, scores)
            if auc < 0.5: auc = roc_auc_score(y_true, -scores)
            auc_results[name].append(auc)

    print("4. Evaluating AUC (Mean ± Std over 5 folds)...")
    final_stats = {}
    for name in ['SVD', 'LightGCN', 'AFRC', 'SORC']:
        mean_auc = np.mean(auc_results[name])
        std_auc = np.std(auc_results[name])
        final_stats[name] = f"{mean_auc:.4f}\pm{std_auc:.4f}"
        print(f"  {name} AUC: {mean_auc:.4f}±{std_auc:.4f}")
        
    return final_stats

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
            f.write(f"Amazon,{a_res['SVD']},{a_res['LightGCN']},{a_res['AFRC']},{a_res['SORC']}\n")
        if m_res:
            f.write(f"MovieLens,{m_res['SVD']},{m_res['LightGCN']},{m_res['AFRC']},{m_res['SORC']}\n")
    print("Saved results to link_prediction_results.txt")
