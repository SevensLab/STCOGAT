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
Create a separated virtual environment:

```shell
conda create -n STCOGAT python=3.10
conda activate STCOGAT
```

### Install packages

Install r-base and mclust packages:

```shell
conda install -c conda-forge r=4.3.0
conda install -c conda-forge r-mclust
```

Install `STCOGAT` from [Github](https://github.com/SevensLab/STCOGAT).

```shell
git clone https://github.com/SevensLab/STCOGAT.git
```

Install `pytorch` package of GPU version and `pyG`. See [Pytorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/2.1.0/index.html) and for detail. Users can choose the corresponding pytorch for other cuda versions. _torch_sparse_, _torch_scatter_, _torch_cluster_ need to be manually downloaded on the [pytorch-geometric](https://pytorch-geometric.com/whl/).

```shell
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

Next,
```Shell
cd STCOGAT
pip install -r requirements.txt
```

Install `jupyter notebook` and set ipykernel.

```shell
conda install jupyter
python -m ipykernel install --user --name STCOGAT --display-name STCOGAT
```

### Use environment.yml
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
We provide source codes for reproducing the STCOGAT analysis in the  `Tutorial` directory.

+ [Mouse olfactory bulb (MOB) clustering (Stereo-seq)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/MOB_Stereo-seq_Clustering.ipynb)
+ [Mouse primary visual cortex (VISp) clustering (STARmap)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/VISp_Clustering.ipynb)
+ [Mouse breast cancer batch effects correction (Visium)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/mouse_breast_Batch_effects.ipynb)
+ [Human BRCA gene embedding (Visium)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/Human_BRCA_Gene.ipynb)

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
