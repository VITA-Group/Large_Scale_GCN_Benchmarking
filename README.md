# Benchmark ScalableGraphLearning

This is an authors' implementation of "Benchmark Scalable Graph Learning" in Pytorch.

Authors: Keyu Duan, Zirui Liu, Wenqing Zheng, Peihao Wang, Kaixiong Zhou, Tianlong Chen, Zhangyang Wang, Xia Hu.

## Introduction

Bag of approaches to train large-scale graphs, including methods based upon
sub-graph sampling, precomputing, and label propagation.

## Requirements

We recommend using anaconda to manage the python environment. To create the environment for our benchmark, please follow the instruction as follows.

```bash
conda create -n $your_env_name
conda activate $your_env_name
```

install pytorch following the instruction on [pytorch installation](https://pytorch.org/get-started/locally/)

```bash
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
```

intall pytorch-geometric following the instruction on [pyg installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

```bash
conda install pyg -c pyg -c conda-forge
```

install the other dependencies

```bash
pip install ogb # package for ogb datasets
pip install texttable # show the running hyperparameters
pip install h5py # for Label Propagation
cd GraphSampling && pip install -v -e . # install our implemented sampler
```

### Our Installation Notes for torch-geometric

What env configs that we tried that have succeeded: Mac/Linux + cuda driver 11.2 + Torch with cuda 11.1 + torch_geometric/torch sparse/etc with cuda 11.1.

What env configs that we tried by did not work: Linux + Cuda 11.1/11.0/10.2 + whatever version of Torch

In the above case when it did work, we adopted the following installation commands, and it automatically downloaded built wheels, and the installation completes within seconds. Installation codes that we adopted on Linux cuda 11.2 that did work:

```bash
  pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
  pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
  pip install torch-geometric
```

**Til now, you should be able to play with all of our implemented models except **Label Propagation**. To run LP, please follow our installation notes.**

### Installation guides for Julia (only required for certain modes of Label propagation, inherited from [C&S](https://github.com/CUAI/CorrectAndSmooth) )

First install Julia and PyJulia, following below instructions or instructions in https://pyjulia.readthedocs.io/en/latest/installation.html#install-julia

#### Installation guide for PyJulia on Linux:

Download Julia from official website, extract to whatever directory on your machine, there will be '/bin' at in the extracted folder.

```bash
export PATH=$PATH:/path-to-yout-extracted-julia/bin
```

After this step, type "julia", then you should be able to see Julia LOGO.

```bash
python3 -m pip install --user julia
```

use python to install

```python
>>> import julia
>>> julia.install()
```

activate julia and install requirements. To activate julia, until you see `julia> `, then type the following lines to install required packages in julia console:

```julia
import Pkg; Pkg.add("LinearMaps")
import Pkg; Pkg.add("Arpack")
import Pkg; Pkg.add("MAT")
```

## Play with our implemented models

To train a scalable graph training model, simply run:

```bash
python main.py --cuda_num=0  --type_model=$type_model --dataset=$dataset
# type_model in ['GraphSAGE', 'FastGCN', 'LADIES', 'ClusterGCN', 'GraphSAINT', 'SGC', 'SIGN', 'SIGN_MLP', 'LP_Adj', 'SAGN', 'GAMLP']
# dataset in ['Flickr', 'Reddit', 'Products', 'Yelp', 'AmazonProducts']
```

To test the throughput and memory usage for a certain model on a dataset, simply add `--debug_mem_speed`

```bash
python main.py --cuda_num=0  --type_model=$type_model --dataset=$dataset --debug_mem_speed
```

To perform the same greedy hyperparemeter search as described in our paper, please run

```bash
python run_HP.py $cuda_num $type_model $dataset
```

For detailed configuration, please refer to `run_HP.py`.

## Reproduce results of EnGCN

Simply run

```bash
# dataset = [Flickr, Reddit, ogbn-products]
bash scripts/$dataset/EnGCN.sh
```

## Some tricks for reducing the memory footprint

1. When using PyG, as illustrated in the [official post](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html), it is recommended to use the **transposed sparse matrix** instead of the edge_index, which can significantly reduce both the memory footprint and the computation overhead. PyG provides a function called [ToSparseTensor](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.ToSparseTensor) to convert the edge index into the transposed sparse matrix.

2. PyG can be used with the mixed precision training or NVIDIA Apex to significantly reduce the memory footprint. Note that the SPMM operater officially support half precision **since the end of August**. You might need to upgrade the torch_sparse package to utilize this new feature.
