import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
from STCOGAT.STCOGAT import STCOGAT
from STCOGAT.Utils import save_model, save_obj
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from STCOGAT.KNNDataset import KNNDataset
from torch.utils.data import DataLoader
import sklearn.neighbors
from scipy import sparse
import warnings
import scipy.sparse as sp
import gc
import os
import pkg_resources
import json
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

INTER_DIM = 450  
EMBEDDING_DIM = 30  
DE_GENES_NUM = 2000 
NUM_LAYERS = 1  
pre_emb_dim = 16
lambda_smooth = 0.05
lambda_contra = 0.1
use_rows_encoder=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train(data: Data,knn_edge_index: torch.LongTensor,highly_variable_index,*,
    number_of_batches: int = 1,max_epoch: int = 400,reduce_interval: int = 30,model_name: str = "",spa_edge_index: torch.LongTensor = None,
    coords: torch.Tensor = None,lr: float = 1e-4,weight_decay: float = 1e-5,save_every: int = 50,
):

    x_full = data.x  # [G, S] gene×spot (float32 CPU)
    model = STCOGAT(
        x_full.shape[0], x_full.shape[1],
        INTER_DIM, EMBEDDING_DIM,
        INTER_DIM, EMBEDDING_DIM,
        pre_emb_dim=pre_emb_dim,
        lambda_smooth=lambda_smooth,
        lambda_contra=lambda_contra,
        num_layers=NUM_LAYERS,
        use_rows_encoder=use_rows_encoder,
    ).to(device)

    x = x_full.clone()
    x = ((x.T - x.mean(dim=1)) / (x.std(dim=1) + 1e-5)).T  # [G,S]
    x = x.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    cur_knn_edge_index = knn_edge_index.to(device)
    cur_knn_edge_index = torch.unique(cur_knn_edge_index, dim=1)

    pbar = tqdm(range(max_epoch), desc="Training", total=max_epoch)
    for epoch in pbar:
        model.train()
        batch_size = max(1, cur_knn_edge_index.size(1) // max(1, number_of_batches))
        loader = mini_batch_knn(cur_knn_edge_index, batch_size)

        for batch in loader:
            knn_batch = batch.T.to(device)  # [2, e_b]

            if use_rows_encoder:
                loss, col_loss, row_loss, contra_loss, smooth_loss, reg = model.calculate_loss(
                    x, knn_batch, data.train_pos_edge_index,
                    highly_variable_index, spa_edge_index, coords
                )
                pbar.set_postfix({
                    "col": f"{col_loss.item():.4f}",
                    "row": f"{row_loss.item():.4f}",
                    "contra": f"{contra_loss.item():.4f}",
                    "smooth": f"{smooth_loss.item():.4f}",
                    "reg": f"{reg.item():.4f}",
                })
            else:
                loss, col_loss, contra_loss, smooth_loss,reg = model.calculate_loss(
                    x, knn_batch, data.train_pos_edge_index,
                    highly_variable_index, spa_edge_index, coords
                )
                pbar.set_postfix({
                    "col": f"{col_loss.item():.4f}",
                    "contra": f"{contra_loss.item():.4f}",
                    "smooth": f"{smooth_loss.item():.4f}",
                    "reg": f"{reg.item():.4f}",
                })

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if reduce_interval > 0 and (epoch + 1) % reduce_interval == 0:
            model.eval()
            with torch.no_grad():
                new_knn_edge_index, _df = model.cols_encoder.reduce_network()
            new_knn_edge_index = torch.unique(new_knn_edge_index.to(device), dim=1)
            cur_knn_edge_index = new_knn_edge_index

        if save_every > 0 and (epoch + 1) % save_every == 0:
            model.eval()
            with torch.no_grad():
                row_embed, col_embed, out_features = model(
                    x, cur_knn_edge_index, data.train_pos_edge_index
                )

            save_obj(cur_knn_edge_index.cpu(),
                     pkg_resources.resource_filename(__name__, r"KNNs/best_new_knn_graph_" + model_name))
            save_obj(col_embed.detach().cpu().numpy(),
                     pkg_resources.resource_filename(__name__, r"Embedding/col_embedding_" + model_name))
            if row_embed is not None:
                save_obj(row_embed.detach().cpu().numpy(),
                         pkg_resources.resource_filename(__name__, r"Embedding/row_embedding_" + model_name))
            save_obj(out_features.detach().cpu().numpy(),
                     pkg_resources.resource_filename(__name__, r"Embedding/out_features_" + model_name))

        gc.collect()

    return model

def build_fused_graph(adata,alpha=0.5,k_spat=8,knn_method="KNN",rad_cutoff=150, 
    prune=True,device=None,):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    A = adata.obsp["distances"]
    A = (A > 0).astype(np.int8)          
    # A = A.multiply(A.T)                 # mutual/AND
    A = A.tocoo()
    expr_ei = torch.tensor(
        np.vstack([A.row, A.col]),
        dtype=torch.long,
        device=device
    )
    expr_ei = torch.unique(expr_ei, dim=1)

    coords = np.asarray(adata.obsm["spatial"])

    if knn_method == "Radius":
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff, n_jobs=-1)
        nbrs.fit(coords)
        distances, indices = nbrs.radius_neighbors(coords, return_distance=True)

        rows, cols = [], []
        for i, (dlist, ilist) in enumerate(zip(distances, indices)):
            for d, j in zip(dlist, ilist):
                if d > 0:
                    rows.append(i); cols.append(j)
        row = np.asarray(rows, dtype=np.int64)
        col = np.asarray(cols, dtype=np.int64)

    else:
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_spat + 1, n_jobs=-1)
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords, return_distance=True)

        dist = distances[:, 1:]
        idx  = indices[:, 1:]

        if prune:
            mask_nonzero = dist > 0
            means = np.nanmean(np.where(mask_nonzero, dist, np.nan), axis=1, keepdims=True)
            stds  = np.nanstd (np.where(mask_nonzero, dist, np.nan), axis=1, keepdims=True)
            cutoff = means + stds
            keep = dist <= cutoff
        else:
            keep = dist > 0

        n, k = idx.shape
        row = np.repeat(np.arange(n, dtype=np.int64), k)
        col = idx.reshape(-1).astype(np.int64)
        keep = keep.reshape(-1)
        row = row[keep]; col = col[keep]

    spat_ei = torch.tensor([row, col], dtype=torch.long, device=device)
    spat_ei = torch.unique(spat_ei, dim=1)

    spat_ei = torch.unique(torch.cat([spat_ei, spat_ei.flip(0)], dim=1), dim=1)

    if alpha <= 1e-8:
        return spat_ei
    if alpha >= 1 - 1e-8:
        return expr_ei

    E = expr_ei.size(1)
    keep_E = int(round(float(alpha) * E))

    if keep_E <= 0:
        return spat_ei

    if keep_E >= E:
        expr_sub = expr_ei
    else:
        r = expr_ei[0].to(torch.int64)
        c = expr_ei[1].to(torch.int64)

        key = (r * 1000003 + c * 9176) & 0x7FFFFFFFFFFFFFFF
        idx = torch.argsort(key)[:keep_E]
        expr_sub = expr_ei[:, idx]

    fused = torch.unique(torch.cat([expr_sub, spat_ei], dim=1), dim=1)
    return fused


def build_spatial_edge_index(obj, k_cutoff=4, device=device):
    coords = np.asarray(obj.obsm['spatial'])
    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=k_cutoff + 1, n_jobs=-1
    )
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords, return_distance=True)
    idx = indices[:, 1:]                     # (n, k_cutoff)

    n, k = idx.shape
    row = np.repeat(np.arange(n, dtype=np.int64), k)
    col = idx.reshape(-1).astype(np.int64)

    edge_index = torch.tensor(
        [row, col],
        dtype=torch.long,
        device=device
    )

    edge_index = torch.unique(edge_index, dim=1)
    return edge_index
  
def _fast_corr_pearson(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    X = X.astype(np.float32, copy=False)

    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, ddof=1, keepdims=False) + 1e-12
    Xn = Xc / std

    n = Xn.shape[0]
    corr = (Xn.T @ Xn) / (n - 1)   # (G,G)
    corr = corr.astype(np.float32, copy=False)
    np.fill_diagonal(corr, -np.inf)
    return corr


def build_coexpres_network(obj,topk: int = 20,min_deg: int = 2,method: str = "pearson",
    positive_only: bool = True,use_abs: bool = False,fallback_allow_nonmutual: bool = True,fallback_allow_negative: bool = False,
):
    X = obj.X
    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    genes = np.asarray(obj.var_names)
    n_spots, G = X.shape
    k = int(min(topk, G - 1))
    min_deg = int(min(min_deg, G - 1))

    if method != "pearson":
        df = pd.DataFrame(X, columns=genes)
        corr = df.corr(method=method).values.astype(np.float32, copy=False)
        np.fill_diagonal(corr, -np.inf)
        df_T = df.T
    else:
        corr = _fast_corr_pearson(X)
        df_T = pd.DataFrame(X.T, index=genes, columns=obj.obs_names)

    if positive_only:
        corr_pos = np.where(corr > 0, corr, -np.inf)
    else:
        corr_pos = corr
    score_pos = np.abs(corr_pos) if use_abs else corr_pos

    nbrs = np.argpartition(-score_pos, kth=k - 1, axis=1)[:, :k]  # (G,k)

    row_idx = np.repeat(np.arange(G), k)
    col_idx = nbrs.reshape(-1)
    w = corr_pos[row_idx, col_idx]
    ok = np.isfinite(w)
    row_idx = row_idx[ok]
    col_idx = col_idx[ok]

    A = sp.csr_matrix(
        (np.ones_like(row_idx, dtype=np.int8), (row_idx, col_idx)),
        shape=(G, G),
    )

    M = A.minimum(A.T).tocoo()
    src = M.row
    dst = M.col
    w_mut = corr[src, dst]
    okm = np.isfinite(w_mut)
    src = src[okm]
    dst = dst[okm]
    w_mut = w_mut[okm]

    adj = np.zeros((G, G), dtype=np.bool_)
    adj[src, dst] = True
    adj[dst, src] = True

    deg = adj.sum(axis=1).astype(np.int32)

    edge_u = [genes[i] for i in src]
    edge_v = [genes[j] for j in dst]
    edge_w = w_mut.astype(np.float32).tolist()

    if min_deg > 0:
        for gi in range(G):
            if deg[gi] >= min_deg:
                continue
            need = int(min_deg - deg[gi])

            if fallback_allow_nonmutual and need > 0:
                cand = nbrs[gi]
                cand_scores = corr_pos[gi, cand]
                valid = np.isfinite(cand_scores)
                cand = cand[valid]
                cand_scores = cand_scores[valid]
                if cand.size > 0:
                    order = np.argsort(-cand_scores)
                    for gj in cand[order]:
                        if need == 0:
                            break
                        if gj == gi:
                            continue
                        if adj[gi, gj]:
                            continue
                        if positive_only and not np.isfinite(corr_pos[gi, gj]):
                            continue

                        adj[gi, gj] = True
                        adj[gj, gi] = True
                        deg[gi] += 1
                        deg[gj] += 1

                        edge_u.append(genes[gi])
                        edge_v.append(genes[gj])
                        edge_w.append(float(corr[gi, gj]))
                        need -= 1

            # 2) 放宽负相关补齐（仍按你的开关 + 2*topk 候选池）
            if fallback_allow_negative and need > 0:
                k2 = int(min(2 * topk, G - 1))
                score_all = np.abs(corr) if use_abs else corr
                cand2 = np.argpartition(-score_all[gi], kth=k2 - 1)[:k2]
                cand2 = cand2[cand2 != gi]
                cand2_scores = score_all[gi, cand2]
                valid2 = np.isfinite(cand2_scores)
                cand2 = cand2[valid2]
                cand2_scores = cand2_scores[valid2]
                order2 = np.argsort(-cand2_scores)
                for gj in cand2[order2]:
                    if need == 0:
                        break
                    if adj[gi, gj]:
                        continue
                    wij = corr[gi, gj]
                    if not np.isfinite(wij):
                        continue

                    adj[gi, gj] = True
                    adj[gj, gi] = True
                    deg[gi] += 1
                    deg[gj] += 1

                    edge_u.append(genes[gi])
                    edge_v.append(genes[gj])
                    edge_w.append(float(wij))
                    need -= 1

    gp = nx.Graph()
    gp.add_nodes_from(genes.tolist())
    gp.add_weighted_edges_from(zip(edge_u, edge_v, edge_w), weight="Weight")

    net = pd.DataFrame({"Source": edge_u, "Target": edge_v, "Weight": edge_w})
    node_feature = df_T.loc[genes]  

    return net, gp, node_feature

def mini_batch_knn(edge_index, batch_size):
    knn_dataset = KNNDataset(edge_index)
    knn_loader = DataLoader(knn_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return knn_loader


def nx_to_pyg_edge_index(G, node_order, device):
    mapping = {n: i for i, n in enumerate(node_order)}
    rows, cols = [], []
    for u, v in G.edges():
        if (u in mapping) and (v in mapping) and (u != v):
            rows.append(mapping[u])
            cols.append(mapping[v])
    if len(rows) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
        edge_index = torch.unique(edge_index, dim=1)
    return edge_index
  
def run_STCOGAT(obj,genet=None, node_feature=None,knn_edge_index=None,
          number_of_batches=1,max_epoch=300, model_name="", save_model_flag = True):

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    node_order = list(node_feature.index)
    genet_edge_index = nx_to_pyg_edge_index(genet, node_order, device)

    genet_edge_index = genet_edge_index.to(device)

    sc.pp.highly_variable_genes(obj,n_top_genes=DE_GENES_NUM)  
    highly_variable_index =  obj.var.highly_variable
    obj = obj[:,node_feature.index]
    highly_variable_index = highly_variable_index[node_feature.index] 
    node_feature.to_pickle(pkg_resources.resource_filename(__name__,r"Embedding/node_features_" + model_name)) 

    x = node_feature.values #gene*spot
    x = torch.tensor(x, dtype=torch.float32).cpu() 
    data = Data(x,edge_index = genet_edge_index)  
    data = train_test_split_edges(data,test_ratio=0.2, val_ratio=0) 

    spa_edge_index = build_spatial_edge_index(obj,k_cutoff=6)
    coords_np = np.asarray(obj.obsm['spatial'])        
    coords = torch.from_numpy(coords_np).float()      
    coords = coords.to(device)  
    model = train(data, knn_edge_index,highly_variable_index, number_of_batches=number_of_batches, max_epoch=max_epoch,
                    model_name=model_name,spa_edge_index=spa_edge_index,coords=coords) 
    
    if save_model_flag:
      save_model(pkg_resources.resource_filename(__name__, r"Models/STCOGAT_" + model_name + ".pt"), model)  
    return model