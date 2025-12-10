import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ss
from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph
import scipy.sparse.linalg as spla
import networkx as nx

def _build_adj_lap(coords, n_neighbors=6, weighted=False, sigma=None, normalize_lap=True, mutual=True):
    """
    Return symmetric adjacency W(csr) and Laplacian L(csr).
    When weighted=True, Gaussian weights are used; if sigma=None, use median of non-zero neighbor distances.
    mutual=True keeps only mutual nearest neighbors (more stable).
    """
    if weighted:
        A = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='distance', include_self=False)
        # Gaussian mapping
        dpos = A.data[A.data > 0]
        if dpos.size == 0:
            raise ValueError("All neighbor distances are zero; check coordinates or n_neighbors.")
        if sigma is None:
            sigma_val = np.median(dpos)
        else:
            sigma_val = float(sigma)
        W = A.tocsr(copy=True)
        W.data = np.exp(-(W.data ** 2) / (2.0 * (sigma_val ** 2) + 1e-12))
    else:
        W = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='connectivity', include_self=False).astype(float).tocsr()
    # Symmetrize
    if mutual:
        W = W.minimum(W.T)   # Mutual nearest neighbors
    else:
        W = 0.5 * (W + W.T)
    W.eliminate_zeros()

    deg = np.array(W.sum(axis=1)).ravel()
    if normalize_lap:
        D_is = ss.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        L = ss.eye(W.shape[0], format='csr') - D_is @ W @ D_is
    else:
        L = ss.diags(deg) - W
    return W.tocsr(), L.tocsr()


def _gft_energy_scores(
    X, L,
    k_low='auto', t=1.0,
    center=True, zscore=True,
    precomputed=None,
):
    """
    Compute GFT low-frequency energy ratios (heat-kernel weighted).
    Support caching: reuse {vals, vecs, Xw, UTX, den} across multiple t for same K.

    Return:
      scores: (G,) energy ratio per gene
      cache:  dict containing {vals, vecs, Xw, UTX, den} for reuse
    """
    if precomputed is None:
        precomputed = {}

    n = L.shape[0]

    # Preprocess expression: centering / z-score
    if 'Xw' in precomputed:
        Xw = precomputed['Xw']
    else:
        Xw = X.copy()
        if center:
            Xw -= Xw.mean(axis=0, keepdims=True)
        if zscore:
            std = Xw.std(axis=0, ddof=1, keepdims=True) + 1e-12
            Xw /= std
        precomputed['Xw'] = Xw

    # ||x||^2
    den = precomputed.get('den')
    if den is None:
        den = (Xw ** 2).sum(axis=0) + 1e-12
        precomputed['den'] = den

    # Low-frequency eigenpairs
    vals = precomputed.get('vals')
    vecs = precomputed.get('vecs')
    if vals is None or vecs is None:
        if isinstance(k_low, str) and k_low == 'auto':
            m = int(min(max(8 * np.ceil(np.sqrt(n)), 10), n - 2))
        else:
            m = int(min(int(k_low), n - 2))
        vals, vecs = spla.eigsh(L.astype(float), k=m, which='SM')
        vals = np.maximum(vals, 0.0)
        precomputed['vals'] = vals
        precomputed['vecs'] = vecs
        precomputed.pop('UTX', None)  # eigenbasis changed → recompute projection

    # Frequency-domain projection U^T X
    UTX = precomputed.get('UTX')
    if UTX is None:
        UTX = vecs.T @ Xw  # [m, G]
        precomputed['UTX'] = UTX

    # Heat kernel weights and accumulate energy
    weights = np.exp(-float(t) * vals)[:, None]         # [m,1]
    num = (weights * (UTX ** 2)).sum(axis=0)            # [G]
    scores = (num / den).ravel()
    return scores, precomputed


def _morans_I_scores(X, W, center=True):
    """
    X: (n_spots x n_genes) dense
    W: (n_spots x n_spots) csr, symmetric, nonnegative
    I_g = (n/Wsum) * (v^T W v)/(v^T v), where v = x - mean(x)
    """
    n = W.shape[0]
    Xw = X.copy()
    if center:
        Xw -= Xw.mean(axis=0, keepdims=True)
    Wsum = float(W.sum())
    WV = W @ Xw
    num = (Xw * WV).sum(axis=0)
    den = (Xw ** 2).sum(axis=0) + 1e-12
    I = (n / (Wsum + 1e-12)) * (num / den)
    return I.ravel()


def select_svg(
    adata,
    N=3000,
    spatial_key='spatial',
    weighted=False,
    sigma=None,
    normalize_lap=True,
    mutual=True,
    k_low='auto', 
    mode='borda',
    center=True, zscore=True,
):
    """
    Single-run (K, t) SVG selection: GFT low-frequency energy ratio + Moran's I → merge by intersection or Borda ranking.
    Depends on: _build_adj_lap, _gft_energy_scores, _morans_I_scores.
    """
    K = 6
    t = 1.0
    # Extract coordinates and expression
    if spatial_key not in adata.obsm_keys():
        raise KeyError(f"{spatial_key} not in adata.obsm_keys().")
    coords = adata.obsm[spatial_key]
    X = adata.X.toarray() if ss.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float64, copy=False)
    genes = np.array(adata.var_names)
    G = X.shape[1]

    # Construct graph and Laplacian (once)
    W, L = _build_adj_lap(
        coords, n_neighbors=K,
        weighted=weighted, sigma=sigma,
        normalize_lap=normalize_lap, mutual=mutual
    )

    # Metric 1: Moran's I
    mor_scores = _morans_I_scores(X, W, center=center)
    mor_order  = np.argsort(-mor_scores)
    mor_rank   = np.empty_like(mor_order); mor_rank[mor_order] = np.arange(1, G+1)

    # Metric 2: GFT low-frequency score
    gft_scores, _cache = _gft_energy_scores(
        X, L, k_low=k_low, t=t,
        center=center, zscore=zscore, precomputed=None
    )
    gft_order = np.argsort(-gft_scores)
    gft_rank  = np.empty_like(gft_order); gft_rank[gft_order] = np.arange(1, G+1)

    # Merge strategies
    if mode == 'intersection':
        gft_topN = set(genes[gft_order[:N]])
        mor_topN = set(genes[mor_order[:N]])
        inter    = list(gft_topN & mor_topN)

        # Maybe fewer than N → fill with Borda
        borda = (G - gft_rank + 1) + (G - mor_rank + 1)
        rest_mask = ~np.isin(genes, inter)
        rest_genes = genes[rest_mask]
        rest_borda = borda[rest_mask]
        rest_sorted = rest_genes[np.argsort(-rest_borda)]
        selected = inter + rest_sorted[:max(0, N - len(inter))].tolist()

    else:  # 'borda'
        borda = (G - gft_rank + 1) + (G - mor_rank + 1)
        order = np.argsort(-borda)
        selected = genes[order[:N]].tolist()

    return selected


def low_pass_enhancement(
    adata,
    ratio_low_freq='infer',
    ratio_high_freq='infer',     # placeholder, no longer used
    ratio_neighbors='infer',
    c=1e-4,
    spatial_info=['array_row', 'array_col'],
    normalize_lap=False,
    inplace=False,
    k_low='auto',
    power=1.0,
    gamma_residual=0.05,
    weighted=True,
    sigma=None,
):
    """
    Spectral-truncated low-pass filtering:
        Z = U_k * diag((1/(1+cλ_k))**power) * (U_k^T X)
            + gamma_residual * (I - U_k U_k^T) X

    - gamma_residual=0 → strict truncation
    - Larger power → stronger denoising
    """

    if ratio_neighbors == 'infer':
        if adata.shape[0] <= 500:
            num_neighbors = 4
        else:
            num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 * ratio_neighbors))

    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)

    x_is_sparse = ss.issparse(adata.X)
    X = adata.X.toarray() if x_is_sparse else np.asarray(adata.X)
    X = X.astype(np.float64, copy=False)
    spot_sums = X.sum(axis=1, keepdims=True)

    def _get_coords(_adata, key):
        if isinstance(key, str) and key in _adata.obsm_keys():
            return _adata.obsm[key]
        elif isinstance(key, (list, tuple)) and set(key) <= set(_adata.obs_keys()):
            return _adata.obs[list(key)].values
        else:
            raise KeyError(f"{key} not found in adata.obsm or adata.obs")

    coords = _get_coords(adata, spatial_info)

    if not weighted:
        A = kneighbors_graph(coords, n_neighbors=num_neighbors, mode='connectivity', include_self=False)
        A = ((A + A.T) > 0).astype(float).tocsr()   # symmetric unweighted
    else:
        A_dist = kneighbors_graph(coords, n_neighbors=num_neighbors, mode='distance', include_self=False)
        dpos = A_dist.data[A_dist.data > 0]
        if dpos.size == 0:
            raise ValueError("All neighbor distances are zero; check coordinates or n_neighbors.")
        if sigma is None:
            sigma_val = np.median(dpos)
        else:
            sigma_val = float(sigma)
        A = A_dist.tocsr(copy=True)
        A.data = np.exp(-(A.data ** 2) / (2.0 * (sigma_val ** 2) + 1e-12))
        A = 0.5 * (A + A.T)  # symmetric weighted

    deg = np.array(A.sum(axis=1)).ravel()
    n = A.shape[0]
    if not normalize_lap:
        L = ss.diags(deg) - A
    else:
        D_is = ss.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        L = ss.eye(n, format='csr') - D_is @ A @ D_is

    if isinstance(k_low, str) and k_low == 'auto':
        k_low_val = int(min(max(8 * np.ceil(np.sqrt(n)), 10), n - 2))
    elif isinstance(k_low, (int, np.integer)) and k_low > 0:
        k_low_val = int(min(k_low, n - 2))
    else:
        if ratio_low_freq == 'infer':
            k_low_val = int(min(max(8 * np.ceil(np.sqrt(n)), 10), n - 2))
        else:
            k_low_val = int(min(np.ceil(np.sqrt(n) * float(ratio_low_freq)), n - 2))

    vals, vecs = spla.eigsh(L.astype(float), k=k_low_val, which='SM')

    h = (1.0 / (1.0 + c * np.maximum(vals, 0.0))) ** float(power)

    # Projection to low-frequency subspace
    Y = vecs.T @ X
    Y *= h[:, None]
    Z_low = vecs @ Y

    # Optional high-frequency residual
    if gamma_residual > 0.0:
        UX = vecs @ (vecs.T @ X)
        Z = Z_low + float(gamma_residual) * (X - UX)
    else:
        Z = Z_low

    Z[Z < 0] = 0.0

    if inplace:
        adata.X = ss.csr_matrix(Z) if x_is_sparse else Z
        return None
    else:
        return pd.DataFrame(Z, index=adata.obs_names, columns=adata.var_names)


def svg(adata, svg_method='gft_top', n_top=3000, csvg=0.0001, smoothing=True):
    assert svg_method in ['gft', 'gft_top', 'seurat', 'seurat_v3']
    if svg_method == 'seurat_v3':
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top)
        adata = adata[:, adata.var['highly_variable']]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata_raw = adata.copy()
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if svg_method == 'seurat':
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
            adata_raw = adata.copy()
        elif svg_method in ['gft', 'gft_top']:
            sel_genes = select_svg(
                adata, N=n_top,
                spatial_key='spatial',
                weighted=True,
                normalize_lap=True,
                mutual=True,
                k_low='auto',
                mode='borda'
            )
            adata = adata[:, sel_genes]
            adata_raw = adata.copy()
            if smoothing:
                low_pass_enhancement(
                    adata, ratio_low_freq=15,
                    c=csvg,
                    spatial_info='spatial',
                    ratio_neighbors=0.3,
                    inplace=True
                )
            adata = adata[:, sel_genes]
    return adata, adata_raw
