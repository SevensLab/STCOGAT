# STCOGAT: Spatial Transcriptomics Analysis via Co-Expression-Aware Graph Attention Autoencoder for Domain Identification and Cross-Slice Integration
STCOGAT includes three key components: data processing, the STCOGAT model (comprising Encoder and Decoder), and downstream analysis. During data processing, raw data is transformed to identifying spatially variable genes (SVGs) and denoised via spectral low-pass filtering to mitigate dropouts. Subsequently, two complementary graphs are constructed: a spot network that fuses spatial proximity with expression similarity, and a gene co-expression network derived from global correlations. The Triple-view encoder aggregates information from these networks to extract comprehensive latent representations for both genes and spots. Two decoders reconstruct the gene expression profiles and gene network respectively to ensure feature fidelity. Notably, an edge-attention mechanism is employed to dynamically prune the spot network, effectively filtering out noisy connections to better reflect true biological heterogeneity. Finally, the learned latent representations are then utilized for downstream tasks, including spatial domain identification, batch effect correction, and multi-slice integration, facilitating a deeper understanding of complex tissue architectures.
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

Install `STCOGAT` from [Github](https://github.com/jxLiu-bio/DeepGFT).

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
+ [Human lymph node gene embedding (Visium)](https://github.com/SevensLab/STCOGAT/blob/master/Tutorial/HLN_geneEmbed.ipynb)

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
