# DSI-3D 

This repository contains the code used for the CBMI 2025 paper "DSI-3D: Differentiable Search Index
for point clouds retrieval"

We extend the Differentiable Search Index (DSI) to accelerate the retrieval phase of of 3D point clouds using the
[GIT](https://arxiv.org/abs/2205.14100.pdf) architecture.

The model is trained to associate point cloud representations, using [LoGG3D-Net](https://github.com/csiro-robotics/LoGG3D-Net/tree/main), with corresponding labels. 
During inference, it generates labels auto-regressively via beam search.

![plot](https://github.com/Chahine-Nicolas/DSI-3D/blob/main/architecture.png?raw=true)

# NEWS

[2025-0X] code release

# Installation

```highlight
conda env create --name DSI_3D --file=environments.yml
```

# Dataset indexing


```highlight
python compute_hierarchical_index.py
```

```highlight
python compute_hilbert_index.py
```


# Checkpoint
coming soon

# Training

You may need to change LABEL_MODE to choose an indexation strategy.

```highlight
source train.sh
```

# Inference

For Naively Structured String identifiers

```highlight
source unit_test_label.sh
```

For Semantically Structured identifiers

```highlight
source unit_test_hierar.sh
```

For Positional Structured identifiers with coordinates interlacing

```highlight
source unit_test_gps.sh
```

For Positional Structured identifiers with Hilbert curve indexing

```highlight
source unit_test_hilbert.sh
```
