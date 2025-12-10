import torch
import numpy as np
import pandas as pd 
import scanpy as sc
from scipy.stats import ranksums
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
import pickle 
import pkg_resources
import os 
alpha  = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsilon = 0.0001

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
        raise KeyError(f"emb_key='{emb_key}' not in adata_integrated.obsm. {list(adata_integrated.obsm_keys())}")
    if batch_column not in adata_integrated.obs.columns:
        raise KeyError(f"batch_column='{batch_column}' not in adata_integrated.obs. {list(adata_integrated.obs.columns)}")

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


def normW(W):
    sum_rows = pd.DataFrame(W.sum(axis=1)) + epsilon
    sum_rows = sum_rows @ sum_rows.T
    sum_rows **= 1/2
    return W / sum_rows

def calculate_propagation_matrix(W, epsilon = 0.0001):
   # device = torch.device("cpu")
    S =  []
    W = normW(W)
    W = torch.tensor(W.values).to(device)
    for index in range(W.shape[0]):
        y = torch.zeros(W.shape[0],dtype=torch.float32).to(device)
        y[index] = 1
        f = y.clone()
        flag = True

        while(flag):
            next_f = (alpha*(W@f) + (1-alpha)*y).to(device)
        
            if torch.linalg.norm(next_f - f) <= epsilon:
                flag = False
            else:
              #  print(torch.linalg.norm(next_f - f))
                f = next_f
        S.append(f)
    return torch.concat(S).view(W.shape)

def propagate_all_genes(W,exp):
    S = calculate_propagation_matrix(W)
    prop_exp = torch.tensor(exp.values).to(device).T
    prop_exp = S @ prop_exp
    prop_norm = S @ torch.ones_like(prop_exp)
    prop_exp /= prop_norm
    prop_exp = pd.DataFrame(prop_exp.T.detach().cpu().numpy(),index = exp.index, columns = exp.columns)
    return prop_exp

def one_step_propagation(W,F):
    W = torch.tensor(normW(W).values, dtype= torch.float32)
    F = torch.tensor(F,dtype= torch.float32)
    prop_exp = (alpha)*W@F + (1-alpha)*F
    prop_norm  = (alpha)*W@torch.ones_like(F) + (1-alpha)*torch.ones_like(F)
    return prop_exp/prop_norm

def add_noise(obj,alpha = 0.0, drop_out = False):
    obj_noise = obj.raw.to_adata()
    #obj_noise.X = (1-alpha) *obj_noise.X + alpha*np.random.randn(*obj.X.shape)
    if drop_out:
        obj_noise.X = obj_noise.X * np.random.binomial(1,(1-alpha),obj.X.shape)
    else:
        obj_noise.X = ((1-alpha) *obj_noise.X + alpha*np.random.randn(*obj.X.shape)).astype(np.float32)
    obj_noise.var["highly_variable"] = True    
    sc.tl.pca(obj_noise, svd_solver='arpack',use_highly_variable = False)
    sc.pp.neighbors(obj_noise,n_pcs=20, n_neighbors=50)
    obj_noise.raw = obj_noise

    return obj_noise

def wilcoxon_enrcment_test(up_sig, down_sig, exp):
    gene_exp = exp.loc[exp.index.isin(up_sig)]
    if down_sig is None:     
        backround_exp = exp.loc[~exp.index.isin(up_sig)]
    else:
        backround_exp = exp.loc[exp.index.isin(down_sig)]
        
    rank = ranksums(backround_exp,gene_exp,alternative="less")[1] # rank expression of up sig higher than backround
    return -1 * np.log(rank)


# ---------------------------
# calculates the signature of the data
#
# returns scores vector of signature calculated per cell
# ---------------------------
def signature_values(exp, up_sig, down_sig=None):
    up_sig = pd.DataFrame(up_sig).squeeze()
    # first letter of gene in upper case
    up_sig = up_sig.apply(lambda x: x[0].upper() + x[1:].lower())
    # keep genes in sig that appear in exp data
    up_sig = up_sig[up_sig.isin(exp.index)]

    if down_sig is not None:
        down_sig = pd.DataFrame(down_sig).squeeze()
        down_sig = down_sig.apply(lambda x: x[0].upper() + x[1:].lower())
        down_sig = down_sig[down_sig.isin(exp.index)]
    
    return exp.apply(lambda cell: wilcoxon_enrcment_test(up_sig, down_sig, cell), axis=0)

def run_signature(obj, up_sig, down_sig=None, umap_flag = True, alpha = 0.9,prop_exp = None):
    """
    Calculate and visualize a propagated signature score for cells in the given object.
    Parameters
    ----------
    obj : AnnData
        The annotated data object containing gene expression matrix and graph data.
    up_sig : list or set
        A collection of genes used to calculate the up-regulated signature score.
    down_sig : list or set, optional
        A collection of genes used to calculate the down-regulated signature score.
        If None, only the up-regulated signature is used. Default is None.
    umap_flag : bool, optional
        If True, generates a UMAP plot colored by the calculated signature score.
        If False, generates a t-SNE plot. Default is True.
    alpha : float, optional
        A parameter controlling the smoothing or propagation factor during signature
        score calculation. Default is 0.9.
    prop_exp : None or other, optional
        An unused parameter placeholder, reserved for future use or extended
        signature propagation functionality.
    Returns
    -------
    np.ndarray
        An array of propagated signature scores, with one score per cell. The
        scores are also stored in obj.obs["SigScore"].
    """

    exp = obj.to_df().T
    graph = obj.obsp["connectivities"].toarray()
    sigs_scores = signature_values(exp, up_sig, down_sig)
    sigs_scores = propagation(sigs_scores, graph)
    obj.obs["SigScore"] = sigs_scores
    # color_map = "jet"
    if umap_flag:
        sc.pl.umap(obj, color=["SigScore"],color_map="magma")
    else:
        sc.pl.tsne(obj, color=["SigScore"],color_map="magma")
    return sigs_scores

def calculate_roc_auc(idents, predict):
    fpr, tpr, _ = roc_curve(idents, predict, pos_label=1)
    return auc(fpr, tpr)

def calculate_aupr(idents, predict):
    return average_precision_score(idents, predict)

def calculate_roc_auc(idents, predict):
    fpr, tpr, _ = roc_curve(idents, predict, pos_label=1)
    return auc(fpr, tpr)

def calculate_aupr(idents, predict):
    return average_precision_score(idents, predict)

# ---------------------------
# Y - scores vector of cells
# W - Adjacency matrix
#
# f_t = alpha * (W * f_(t-1)) + (1-alpha)*Y
#
# returns f/f1
# ---------------------------
def propagation(Y, W):
    W = normW(W)
    f = np.array(Y)
    Y = np.array(Y)
   # f2 = calculate_propagation_matrix(W) @ Y

    W = np.array(W.values)
    
    Y1 = np.ones(Y.shape, dtype=np.float64)
    f1 = np.ones(Y.shape, dtype=np.float64)
    flag = True

    while(flag):
        next_f = alpha*(W@f) + (1-alpha)*Y
        next_f1 = alpha*(W@f1) + (1-alpha)*Y1
    
        if np.linalg.norm(next_f - f) <= epsilon and np.linalg.norm(next_f1 - f1) <= epsilon:
            flag = False
        else:
            #print(np.linalg.norm(next_f - f))
            #print(np.linalg.norm(next_f1 - f1))
            f = next_f
            f1 = next_f1
   # return f1,f2
    return np.array(f/f1) 


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