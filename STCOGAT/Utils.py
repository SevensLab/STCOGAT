import torch
import numpy as np
import pandas as pd 
import pickle 
import pkg_resources
import os 
import scanpy as sc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')
from anndata import AnnData
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri,numpy2ri

ro.r.source('./STCOGAT/BatchKL.R')
def BatchKL(adata_integrated, batch_column="batch", emb_key="embed", n_cells=100):
    if emb_key not in adata_integrated.obsm_keys():
        raise KeyError(f"emb_key='{emb_key}' 不在 adata_integrated.obsm 里。现有键：{list(adata_integrated.obsm_keys())}")
    if batch_column not in adata_integrated.obs.columns:
        raise KeyError(f"batch_column='{batch_column}' 不在 adata_integrated.obs 列里。现有列：{list(adata_integrated.obs.columns)}")

    emb_np = adata_integrated.obsm[emb_key]          # numpy array / matrix
    meta_df = adata_integrated.obs                   # pandas DataFrame

    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        meta_r = ro.conversion.py2rpy(meta_df)
        emb_r  = ro.conversion.py2rpy(emb_np)
        KL = ro.r["BatchKL"](meta_r, emb_r, n_cells=n_cells, batch=batch_column)

    try:
        print("BatchKL =", float(KL[0]) if len(KL) > 0 else KL)
    except Exception:
        print("BatchKL =", KL)

    return KL


ro.r('options(warn = -1)')
mclust = importr('mclust', suppress_messages=True)
base = importr('base')
mclust = importr('mclust', suppress_messages=True)
utils = importr('utils')

def mclust_R(adata: AnnData, num_cluster: int, modelNames='EEE', used_obsm='embed',random_seed=2025, obs_key='mclust') -> AnnData:
    np.random.seed(random_seed)
    robjects.r(f'set.seed({random_seed})')

    if used_obsm is not None and used_obsm in adata.obsm:
        X = adata.obsm[used_obsm]
    else:
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    with localconverter(default_converter + numpy2ri.converter):
        r_mat = robjects.conversion.py2rpy(X)
    ro.r('''
    suppress_stdout <- function(expr) {
        tf <- tempfile()
        sink(tf)
        on.exit({
            sink()
            unlink(tf)
        })
        force(expr)
    }
    ''')

    ro.globalenv['r_mat'] = r_mat
    ro.r(f'''
    fit_result <- suppress_stdout(
        mclust::Mclust(r_mat, G={num_cluster}, modelNames="{modelNames}")
    )
    ''')
    res = ro.r['fit_result']

    labels = np.array(res[-2]).astype(int)
    adata.obs[obs_key] = labels
    adata.obs[obs_key] = adata.obs[obs_key].astype('category')
    return adata


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_model(path, model):
    torch.save(model.state_dict(), path)


def load_embeddings(proj_name):
    '''
    Loads the embeddings and gene expression data for a given project.

    Args:
        proj_name (str): The name of the project.

    Returns:
        tuple: A tuple containing:
            - embedded_genes (np.ndarray): Learned gene embeddings.
            - embedded_cells (np.ndarray): Learned cell embeddings.
            - node_features (pd.DataFrame): Original gene expression matrix.
            - out_features (np.ndarray): Reconstructed gene expression matrix.
    '''
    path = pkg_resources.resource_filename(__name__,r"./Embedding/row_embedding_" + proj_name+".pkl")
    if os.path.exists(path):
        embeded_genes = load_obj(pkg_resources.resource_filename(__name__,r"./Embedding/row_embedding_" + proj_name))
    else:
        embeded_genes = None
    embeded_cells = load_obj(pkg_resources.resource_filename(__name__,r"./Embedding/col_embedding_" + proj_name))
    node_features = pd.read_pickle(pkg_resources.resource_filename(__name__,r"./Embedding/node_features_" + proj_name))
    out_features = load_obj(pkg_resources.resource_filename(__name__,r"./Embedding/out_features_" + proj_name))
    return embeded_genes, embeded_cells, node_features, out_features

def create_reconstructed_obj(node_features, out_features, orignal_obj=None):
  '''
    Creates an AnnData object from reconstructed gene expression data, normalizes it, and computes PCA, neighbors, clustering, and UMAP.

    Args:
        node_features (pd.DataFrame): The original gene expression matrix with genes as columns and cells as rows.
        out_features (np.ndarray): The reconstructed gene expression matrix.
        original_obj (AnnData, optional): The original AnnData object, if available, to copy cell metadata (obs) from. Defaults to None.

    Returns:
        AnnData: An AnnData object containing the reconstructed gene expression data.
    '''
  embd = pd.DataFrame(out_features,index=node_features.columns[:out_features.shape[0]], columns=node_features.index)

  embd = (embd - embd.min()) / (embd.max() - embd.min())

  adata = sc.AnnData(embd)
  if not orignal_obj is None:
    adata.obs = orignal_obj.obs[:embd.shape[0]]
    adata.obsm['spatial'] = orignal_obj.obsm['spatial']
  return adata