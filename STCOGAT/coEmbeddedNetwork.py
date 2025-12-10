import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

def create_reconstructed_obj(node_features, out_features, orignal_obj=None):
  '''
    Creates an AnnData object from reconstructed gene expression data.

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



    
