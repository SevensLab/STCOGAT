import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
from STCOGAT.STCOGAT import STCOGAT
from STCOGAT.Utils import save_model, save_obj
import torch
from torch_geometric.utils import  convert
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from STCOGAT.KNNDataset import KNNDataset, CellDataset
from torch.utils.data import DataLoader
import sklearn.neighbors
from scipy.spatial.distance import cdist
from scipy import sparse
import warnings
import gc
import pkg_resources
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

INTER_DIM = 256  
EMBEDDING_DIM = 30  
pre_emb_dim = 16
NUM_LAYERS = 0  
DE_GENES_NUM = 2000  
lambda_cluster = 0.1
lambda_smooth = 0.5
lambda_contra = 1
num_clusters = 7
use_rows_encoder=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def test_recon(model,x, data, knn_edge_index):

    """
    Evaluate model reconstruction performance on test edges.
    Args:
      model (torch.nn.Module): Trained STCOGAT model.
      x (torch.Tensor): Input features for the nodes.
      data (torch_geometric.data.Data): Graph data object containing positive and negative edges.
      knn_edge_index (torch.Tensor): k-NN graph edges for the rows.
    Returns:
      float: AUC score of edge reconstruction.
    """
    model.eval()
    with torch.no_grad():
        embbed_rows, _, _ = model(x, knn_edge_index, data.train_pos_edge_index)
    return model.test(embbed_rows, data.test_pos_edge_index, data.test_neg_edge_index)

def crate_knn_batch(knn,idxs,k=15):
  idxs = idxs.cpu().numpy()
  adjacency_matrix = torch.tensor(knn[idxs][:,idxs].toarray())
  row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
  knn_edge_index = torch.stack((row_indices, col_indices))
  knn_edge_index = torch.unique(knn_edge_index, dim=1)
  return knn_edge_index.to(device)

def train(data, loader, highly_variable_index,number_of_batches=5 ,
          max_epoch = 500, rduce_interavel = 30,model_name="",spa_edge_index=None,coords=None):
    x_full = data.x.clone()
    model = STCOGAT(x_full.shape[0], x_full.shape[1], INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM,pre_emb_dim = pre_emb_dim,
                      lambda_cluster=lambda_cluster,lambda_smooth=lambda_smooth,lambda_contra=lambda_contra,
                      num_clusters=num_clusters,num_layers=NUM_LAYERS,use_rows_encoder=use_rows_encoder).to(device)
    x = x_full.clone()
    x = ((x.T - (x.mean(axis=1)))/ (x.std(axis=1)+ 0.00001)).T
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    best_auc = 0.5
    concat_flag = False
    pbar = tqdm(range(max_epoch), desc="Training", total=max_epoch)
    for epoch in pbar:
        concat_flag = False
        for _,batch in enumerate(loader):
            model.train()
            knn_edge_index = batch.T.to(device)
            if knn_edge_index.shape[1] == loader.dataset.edge_index.shape[0] // number_of_batches :
                loss,recon_total,loss_cluster = model.calculate_loss(x.clone().to(device), knn_edge_index.to(device),
                                                                data.train_pos_edge_index,highly_variable_index,spa_edge_index,coords)
                loss = loss + loss_cluster
                pbar.set_postfix({
                    "recon": f"{recon_total.item():.4f}",
                    "clust": f"{loss_cluster.item():.4f}",
                })

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    row_embed, col_embed, out_features = model(x.to(device),knn_edge_index.to(device), data.train_pos_edge_index)
            else:
              concat_flag = True
            gc.collect()
            torch.cuda.empty_cache()
        new_knn_edge_index, _ = model.cols_encoder.reduce_network()  
        if concat_flag:
            new_knn_edge_index = torch.concat([new_knn_edge_index,knn_edge_index], axis=-1)
            knn_edge_index = new_knn_edge_index
        if (epoch+1) % rduce_interavel == 0:
            loader = mini_batch_knn(new_knn_edge_index, new_knn_edge_index.shape[1] // number_of_batches)
        if epoch%10 == 0:
            knn_edge_index = list(loader)[0].T.to(device)

        auc, ap = test_recon(model, x.to(device), data, knn_edge_index)
        if auc > best_auc:
            best_auc = auc

        save_obj(knn_edge_index.cpu(),pkg_resources.resource_filename(__name__, r"KNNs/best_new_knn_graph_" + model_name))
        save_obj(col_embed.cpu().detach().numpy(), pkg_resources.resource_filename(__name__,r"Embedding/col_embedding_" + model_name))
        if row_embed is not None:
            save_obj(row_embed.cpu().detach().numpy(), pkg_resources.resource_filename(__name__,r"Embedding/row_embedding_" + model_name))
        save_obj(out_features.cpu().detach().numpy(),  pkg_resources.resource_filename(__name__,r"Embedding/out_features_" + model_name))
    # print(f"Best Gene Network AUC: {best_auc}")
    return model

def build_fused_graph(
    obj,
    alpha: float = 0.5,
    k_spat: int = 8,
    rad_cutoff: float = 150,
    knn_method: str = 'KNN',
    prune: bool = True,
) -> 'torch.LongTensor':

    dist_expr = obj.obsp['distances'].toarray()
    adj_expr = (dist_expr > 0).astype(int)
    adj_expr = adj_expr * adj_expr.T 

    coords = np.array(obj.obsm['spatial'])
    if knn_method == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff, n_jobs=-1)
        nbrs.fit(coords)
        distances, indices = nbrs.radius_neighbors(coords, return_distance=True)
    else:
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_spat + 1, n_jobs=-1)
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        if prune:
            mask_nonzero = distances > 0
            means = np.nanmean(np.where(mask_nonzero, distances, np.nan), axis=1, keepdims=True)
            stds  = np.nanstd (np.where(mask_nonzero, distances, np.nan), axis=1, keepdims=True)
            cutoff = means + stds
            distances[distances > cutoff] = 0
    
    n = coords.shape[0]
    adj_spat = np.zeros((n, n), dtype=float)
    for i, (dlist, nbrs_idx) in enumerate(zip(distances, indices)):
        for dist, j in zip(dlist, nbrs_idx):
            if dist > 0:
                adj_spat[i, j] = 1.0
    
    adj_spat = np.maximum(adj_spat, adj_spat.T)

    fused = alpha * adj_expr + (1 - alpha) * adj_spat

    G = nx.from_numpy_array(fused)
    data = convert.from_networkx(G)
    return data.edge_index.to(device)  


def build_spatial_edge_index(obj, k_cutoff=6, device=device):
    coords = np.asarray(obj.obsm['spatial'])
    
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1,n_jobs=-1)
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords, return_distance=True)
    
    row, col = [], []
    for i, neigh in enumerate(indices):
        for j in neigh:
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long, device=device)

    edge_index = torch.unique(edge_index, dim=1)
    return edge_index
  

def build_coexpres_network(
    obj,
    cut=0.2,
    percentile=0,
    method='pearson',
    spatial_decay=False,
    lambda_=10,
    verbose=False
):

    if isinstance(obj.X, sparse.spmatrix):
        X = obj.X.todense()
    else:
        X = obj.X.copy()
    df = pd.DataFrame(X, columns=obj.var_names)
    if verbose:
        print(f"[INFO] use {method} compute correlations, total {df.shape[1]} genes, {df.shape[0]} spots")
    corr = df.corr(method=method)
    corr_values = corr.copy()
    if spatial_decay:
        if 'spatial' not in obj.obsm:
            raise ValueError("Not found obj.obsm['spatial'] in AnnData")
        coords = np.asarray(obj.obsm['spatial'])
        df_gene = df.T  
        weighted_centroids = np.array([
            np.average(coords, axis=0, weights=expr) if np.sum(expr) > 0 else np.zeros(coords.shape[1])
            for expr in df_gene.values
        ])

        dist_matrix = cdist(weighted_centroids, weighted_centroids, metric='euclidean')
        decay_matrix = np.exp(-dist_matrix / lambda_)
        corr_values.values[:] = corr_values.values * decay_matrix
    if percentile != 0:
        valid_vals = corr_values.values.flatten()
        valid_vals = valid_vals[~np.isnan(valid_vals)]
        threshold = np.percentile(valid_vals, percentile)
    else:
        threshold = cut
    sources, targets = np.where((corr_values.values >= threshold) & (np.ones_like(corr_values.values) - np.eye(corr_values.shape[0]) > 0))
    weights = corr_values.values[sources, targets]
    gene_names = corr.columns.to_numpy()
    net = pd.DataFrame({
        "Source": gene_names[sources],
        "Target": gene_names[targets],
        "Weight": weights
    })
    gp = nx.from_pandas_edgelist(net, source="Source", target="Target", edge_attr="Weight")
    node_feature = df.T.loc[list(gp.nodes)]
    return net, gp, node_feature

def mini_batch_knn(edge_index, batch_size):
    knn_dataset = KNNDataset(edge_index)
    knn_loader = DataLoader(knn_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return knn_loader


def mini_batch_cells(x,edge_index, batch_size):
    cell_dataset = CellDataset(x, edge_index)
    cell_loader = DataLoader(cell_dataset,batch_size=batch_size, shuffle=False, drop_last=True)
    return cell_loader
  

def nx_to_pyg_edge_index(G, mapping=None):
    G = G.to_directed() if not nx.is_directed(G) else G
    if mapping is None:  
       mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long).to(device)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
    return edge_index, mapping

  
def run_STCOGAT(obj,genet=None, node_feature=None,knn_edge_index=None,
          number_of_batches=3,max_epoch=150, model_name="", save_model_flag = False):

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    genet_edge_index, _ = nx_to_pyg_edge_index(genet) 
    genet_edge_index = genet_edge_index.to(device)

    sc.pp.highly_variable_genes(obj,n_top_genes=DE_GENES_NUM) 
    highly_variable_index =  obj.var.highly_variable
    obj = obj[:,node_feature.index]
    loader = mini_batch_knn(knn_edge_index, knn_edge_index.shape[1] // number_of_batches)
    highly_variable_index = highly_variable_index[node_feature.index] 
    node_feature.to_pickle(pkg_resources.resource_filename(__name__,r"Embedding/node_features_" + model_name))  

    x = node_feature.values
    x = torch.tensor(x, dtype=torch.float32).cpu()  
    data = Data(x,edge_index = genet_edge_index)  
    data = train_test_split_edges(data,test_ratio=0.2, val_ratio=0)  

    spa_edge_index = build_spatial_edge_index(obj,k_cutoff=6)
    coords_np = np.asarray(obj.obsm['spatial'])        
    coords = torch.from_numpy(coords_np).float()       
    coords = coords.to(device)  
    model = train(data, loader, highly_variable_index, number_of_batches=number_of_batches, max_epoch=max_epoch,
                    rduce_interavel=30,model_name=model_name,spa_edge_index=spa_edge_index,coords=coords)  
    if save_model_flag:
      save_model(pkg_resources.resource_filename(__name__, r"Models/STCOGAT_" + model_name + ".pt"), model)  
    return model