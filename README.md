# STCOGAT: Spatial Transcriptomics Analysis via Co-Expression-Aware Graph Attention Autoencoder for Domain Identification
STCOGAT constructs two networks: one is a gene co-expression network based on global gene correlations, while the other is a spot network that fuses spatial adjacency matrix derived from k-nearest neighbors (KNN) and the adjacency matrix representing expression similarity. First, STCOGAT employs a triple-view encoder to aggregate information from these two networks and the gene expression matrix to generate initial representations. Next, STCOGAT uses a graph attention encoder to generate gene embeddings (gene view). It also uses a graph attention encoder to model the spatial context on the spot graph, generating spot embeddings (spot view). Finally, STCOGAT's dual decoders reconstruct the gene network and gene expression profiles from the gene embeddings and spot embeddings, respectively. In addition, the edge attention mechanism dynamically prunes the spot graph to filter out noisy connections. STCOGAT also applies a multilayer perceptron (MLP) encoder to encode the intrinsic features of the spots, then concatenates both to generate the final spot embeddings. The learned latent representations are then utilized for downstream tasks, including spatial domain identification, batch effect correction, and multi-slice integration, facilitating the study of tissue heterogeneity.
![STCOGAT](https://github.com/SevensLab/STCOGAT/blob/master/flowchart.png)
## OS requirements

`STCOGAT` can run on Linux and Windows. The package has been tested on the following systems:

- Linux: Ubuntu 25.04, NVIDIA GeForce RTX 5090 D, NVIDIA GeForce RTX 4090, CUDA 12.8
- Windows: Windows 10, NVIDIA GeForce RTX 3080 Ti, CUDA 12.6

## Installation Guide

### Create a virtual environment

Users can install `anaconda` by following this tutorial if there is no [Anaconda](https://www.anaconda.com/).
We recommend creating a conda environment with Python 3.10.

```shell
conda create -n STCOGAT python=3.10
conda activate STCOGAT
```

### Install packages

Install r-base and mclust packages:

```shell
conda install -c conda-forge r=4.3.0 r-mclust
```

Install `STCOGAT` from [Github](https://github.com/SevensLab/STCOGAT).

```shell
git clone https://github.com/SevensLab/STCOGAT.git
```

Install `pytorch` and  `pyG` according to your own CUDA version. Please follow the official PyTorch and PyG installation instructions.

For example, for the CUDA 12.8 environment used in our experiments:

```shell
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

Then install the remaining dependencies:
```Shell
pip install -r requirements.txt
```

Install `jupyter notebook` and set ipykernel.

```shell
conda install jupyter
python -m ipykernel install --user --name STCOGAT --display-name STCOGAT
```

### Use environment.yml
The file `environment.yml` records the environment used in our experiments. It is provided for reproducibility, but users with different CUDA versions are encouraged to follow the installation instructions above rather than directly using this environment file.

``` shell
git clone https://github.com/SevensLab/STCOGAT.git
cd STCOGAT
conda env create -f environment.yml
conda activate STCOGAT
```
If conda fails to install the CUDA version of PyTorch, please install it manually:

```shell
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

## Tutorial
The `Tutorial/` folder contains example notebooks for reproducing the main STCOGAT analyses. These notebooks already provide complete workflows, including data loading, preprocessing, graph construction, model training, embedding extraction, clustering, metric calculation, and visualization.

+ [Mouse olfactory bulb (MOB) clustering (Stereo-seq)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/MOB_Stereo-seq_Clustering.ipynb)
+ [Mouse primary visual cortex (VISp) clustering (STARmap)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/VISp_Clustering.ipynb)
+ [Mouse breast cancer batch effects correction (Visium)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/mouse_breast_Batch_effects.ipynb)
+ [Human BRCA gene embedding (Visium)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/Human_BRCA_Gene.ipynb)

### Workflow overview

A typical STCOGAT analysis follows the steps below:

1. Download the corresponding dataset from the data link below.
2. Place the processed `.h5ad` files under the `data/` directory.
3. Select spatially variable genes using the preprocessing functions.
4. Construct the gene co-expression graph and fused spot graph.
5. Train STCOGAT using `run_STCOGAT`.
6. Load the learned embeddings using `load_embeddings`.
7. Perform clustering with `mclust_R`.
8. Evaluate clustering or integration performance and generate spatial visualizations.

The tutorial notebooks provide executable examples for these steps.
### example
```python
import scanpy as sc
import STCOGAT
from STCOGAT.Utils import *
from STCOGAT.data_preprocess import *

# Load AnnData
adata = sc.read_h5ad("./data/example.h5ad")

# Select spatially variable genes
adata, adata_raw = svg(adata, svg_method="seurat_v3", n_top=2000)

# Build gene graph and spot graph
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=15)
net, genet, node_feature = STCOGAT.build_coexpres_network(adata, topk=20)
knn_edge_index = STCOGAT.build_fused_graph(
    adata,
    alpha=0.5,
    k_spat=6,
    knn_method="KNN",
    prune=True
)

# Train STCOGAT
STCOGAT.run_STCOGAT(
    adata,
    genet,
    node_feature,
    knn_edge_index,
    max_epoch=300,
    model_name="example",
    save_model_flag=True
)

# Load embeddings
_, embedded_cells, node_features, out_features = STCOGAT.load_embeddings("example")
adata.obsm["embed"] = embedded_cells

# Clustering
adata = STCOGAT.mclust_R(
    adata,
    num_cluster=7,
    modelNames="EEE",
    used_obsm="embed",
    random_seed=2026,
    obs_key="mclust_labels"
)
```

## Compared tools
Tools that are compared include:

- [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)
- [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
- [SEDR](https://github.com/JinmiaoChenLab/SEDR/)
- [DeepST](https://github.com/JiangBioLab/DeepST)
- [GraphST](https://github.com/JinmiaoChenLab/GraphST)
- [STAGATE](https://github.com/zhanglabtools/STAGATE)
- [SpaceFlow](https://github.com/hongleir/SpaceFlow)
- [DeepGFT](https://github.com/jxLiu-bio/DeepGFT)
- [STMGraph](https://github.com/binbin-coder/STMGraph)

## Download data
The datasets used in this paper can be downloaded from [here](https://drive.google.com/drive/folders/1m4QlemN5GmKdR1gJHaUl_NMElKI-GNOq?usp=sharing)
