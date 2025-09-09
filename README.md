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

# Installation of LoGG3D-Net

DSI-3D use [LoGG3D-Net](https://github.com/csiro-robotics/LoGG3D-Net/tree/main) so we also need to reproduce their environment.
Our implementation was developed and tested on the Jean Zay HPC cluster.
Since installing torchsparse-1.4.0 can be challenging with admin rights, we relied on an existing Jean Zay module (pytorch-gpu/py3/1.10.1) to pre-compute the LoGG3D-Net features. 

```highlight
module load pytorch-gpu/py3/1.10.1
```

```highlight
python evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 0 \
       --checkpoint_name '/kitti_10cm_loo/2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth' \
       --skip_time 30 
```

```highlight
python evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 2 \
       --checkpoint_name '/kitti_10cm_loo/2021-09-14_05-55-20_3n24h_Kitti_v10_q29_10s2_262448.pth' \
       --skip_time 30   
```


```highlight
python evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 5 \
       --checkpoint_name '/kitti_10cm_loo/2021-09-14_06-11-58_3n24h_Kitti_v10_q29_10s5_262449.pth' \
       --skip_time 30      
```


```highlight
python evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 6 \
       --checkpoint_name '/kitti_10cm_loo/2021-09-14_06-43-47_3n24h_Kitti_v10_q29_10s6_262450.pth' \
       --skip_time 30        
```

```highlight
python evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 7 \
       --checkpoint_name '/kitti_10cm_loo/2021-09-14_08-34-46_3n24h_Kitti_v10_q29_10s7_262451.pth' \
       --skip_time 30    
```
```highlight
python evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 8 \
       --checkpoint_name '/kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth' \
       --skip_time 30
```

For the sequence 22, we use the checkpoint 2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth as:
```highlight
python evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 22 \
       --checkpoint_name '/kitti_10cm_loo/2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth' \
       --skip_time 30  
```




# Dataset

DSI-3D follows the same dataset setup as LoGG3D-Net
Download the [KITTI Odometry datasets.](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
Update the dataset paths in LoGG3D-Net/config/eval_config.py.

After downloading, rename the pose files (00.txt, 02.txt, etc.) to poses.txt and place each one inside its corresponding sequence folder (00/, 02/, …).


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
