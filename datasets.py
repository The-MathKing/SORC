import os
import zipfile
import json
import gzip
import numpy as np
import scipy.sparse as sp
import networkx as nx

CYCLE_AWARE_PATH = "/Users/aryanpadarthi/Cycle-Aware"

def load_amazon_vg():
    """
    Loads the Amazon Video Games 5-core dataset as a bipartite graph.
    Returns (sp.csr_matrix, num_users)
    """
    path = os.path.join(CYCLE_AWARE_PATH, "amazon_vg.json.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
        
    user_map = {}
    item_map = {}
    rows, cols, data = [], [], []
    
    with gzip.open(path, 'rt') as f:
        for line in f:
            js = json.loads(line)
            u = js.get('reviewerID')
            i = js.get('asin')
            r = js.get('overall')
            if u and i and r is not None and r >= 4.0:
                if u not in user_map: user_map[u] = len(user_map)
                if i not in item_map: item_map[i] = len(item_map)
                rows.append(user_map[u])
                cols.append(item_map[i])
                data.append(1)
                
    num_users = len(user_map)
    cols = [c + num_users for c in cols] # Shift item indices
    
    # Make symmetric
    n = max(max(rows), max(cols)) + 1
    
    # We want a symmetric undirected unweighted graph
    sym_rows = rows + cols
    sym_cols = cols + rows
    sym_data = data + data
    
    A = sp.csr_matrix((sym_data, (sym_rows, sym_cols)), shape=(n, n))
    A.data = np.ones_like(A.data) # ensure unweighted
    return A, num_users

def load_ml1m():
    """
    Loads MovieLens-1M as a bipartite graph.
    Returns (sp.csr_matrix, num_users)
    """
    path = os.path.join(CYCLE_AWARE_PATH, "ml-1m.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
        
    rows, cols, data = [], [], []
    num_users = 6040 + 10 
    
    with zipfile.ZipFile(path, 'r') as z:
        with z.open('ml-1m/ratings.dat') as f:
            for line in f:
                parts = line.decode('utf-8').strip().split('::')
                if len(parts) >= 3:
                    u = int(parts[0]) - 1
                    i = int(parts[1]) - 1 + num_users
                    r = int(parts[2])
                    if r >= 4:
                        rows.append(u)
                        cols.append(i)
                        data.append(1)
                        
    n = max(max(rows), max(cols)) + 1
    sym_rows = rows + cols
    sym_cols = cols + rows
    sym_data = data + data
    
    A = sp.csr_matrix((sym_data, (sym_rows, sym_cols)), shape=(n, n))
    A.data = np.ones_like(A.data) # ensure unweighted
    return A, num_users
