# DSI-3D 

This repository contains the code used for the CBMI 2025 paper "DSI-3D: Differentiable Search Index
for point clouds retrieval"

We adapted the Differentiable Search Index to accelerate the retrieval phase of of 3D point cloud using GIT 
![GIT]([https://arxiv.org/pdf/2205.14100)   architecture.

![plot](https://github.com/Chahine-Nicolas/DSI-3D/blob/main/architecture.png?raw=true)

# NEWS

[2025-0X] code release

# Installation
conda create -n DSI_3D python=3.7
conda activate gd-DSI_3D

# Dataset indexing

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
