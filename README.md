# DSI-3D 

This repository contains the code used for the CBMI 2025 paper "DSI-3D: Differentiable Search Index
for point clouds retrieval". Our implementation was developed and tested on the Jean Zay HPC cluster.

We extend the Differentiable Search Index (DSI) to accelerate the retrieval phase of of 3D point clouds using the
[GIT](https://arxiv.org/abs/2205.14100.pdf) architecture.

The model is trained to associate point cloud representations, using [LoGG3D-Net](https://github.com/csiro-robotics/LoGG3D-Net/tree/main), with corresponding labels. 
During inference, it generates labels auto-regressively via beam search.

![plot](https://github.com/Chahine-Nicolas/DSI-3D/blob/main/architecture.png?raw=true)

# NEWS

[2025-0X] code release

# Installation of the DSI-3D environment

```highlight
module load anaconda-py3/2023.03
conda create -y -n DSI_3D python=3.10
conda activate DSI_3D
```

```highlight
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda cudatoolkit
conda install transformers
export PYTHONUSERBASE=/gpfswork/rech///install/sandbox
pip install --user bitsandbytes
pip install --user git+https://github.com/huggingface/accelerate.git
```

GIT requierements
```highlight
pip install -r requirements.txt
```

```highlight
pip install --user tensorboardX tensorboard
pip install --user easydict
pip install --user nuscenes-devkit spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython
pip install --user pycocotools SharedArray terminaltables 
pip install --user torchvision
pip install --user timm
pip install --user einops
#pip install --user torch-scatter
pip install --user torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install --user hostlist
pip install --user hilbertcurve
```

# Dataset

DSI-3D follows the same dataset setup as LoGG3D-Net
Download the [KITTI Odometry datasets.](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
Update the dataset paths in LoGG3D-Net/config/train_config.py and LoGG3D-Net/config/eval_config.py.

After downloading, rename the pose files (00.txt, 02.txt, etc.) to poses.txt and place each one inside its corresponding sequence folder (00/, 02/, …).


## Building the sequence 22

We build a new "KITTI", named 22, while respecting the previous formalism of the kitti format.

Start by creating the repository  
```highlight
cd  **kitti_dir_path**
mkdir 22
cp -r /00/velodyne /22/velodyne
```

For the following sequence, we add the cumulative length of the current sequence 22
```highlight
shopt -s nullglob

# Loop over all .bin files in 02/velodyne
for file in 02/velodyne/*.bin; do
    base_value=$(basename "$file" .bin)    
    base_value=${base_value#0}             
    base_value=${base_value:-0}        
    new_value=$((10#$base_value + 4541)) 
    new_name=$(printf "%06d.bin" "$new_value")
    cp "$file" "22/velodyne/$new_name"   
done

shopt -u nullglob
```

```highlight
shopt -s nullglob

for file in 05/velodyne/*.bin; do
    base_value=$(basename "$file" .bin)    
    base_value=${base_value#0}             
    base_value=${base_value:-0}        
    new_value=$((10#$base_value + 9202)) 
    new_name=$(printf "%06d.bin" "$new_value")
    cp "$file" "22/velodyne/$new_name"   
done

shopt -u nullglob
```
```highlight
shopt -s nullglob

for file in 06/velodyne/*.bin; do
    base_value=$(basename "$file" .bin)    
    base_value=${base_value#0}             
    base_value=${base_value:-0}        
    new_value=$((10#$base_value + 11963)) 
    new_name=$(printf "%06d.bin" "$new_value")
    cp "$file" "22/velodyne/$new_name"   
done

shopt -u nullglob
```

```highlight
shopt -s nullglob

for file in 07/velodyne/*.bin; do
    base_value=$(basename "$file" .bin)    
    base_value=${base_value#0}             
    base_value=${base_value:-0}        
    new_value=$((10#$base_value + 13064)) 
    new_name=$(printf "%06d.bin" "$new_value")
    cp "$file" "22/velodyne/$new_name"   
done

shopt -u nullglob
```

```highlight
shopt -s nullglob

for file in 08/velodyne/*.bin; do
    base_value=$(basename "$file" .bin)    
    base_value=${base_value#0}             
    base_value=${base_value:-0}        
    new_value=$((10#$base_value + 14165)) 
    new_name=$(printf "%06d.bin" "$new_value")
    cp "$file" "22/velodyne/$new_name"   
done

shopt -u nullglob
```

You may check that your last bin file is named "018235.bin" 
and that 
```highlight
ls -1 | wc -l
```
return 18236

You need to update the revisit file to add sequence 22
```highlight
python updat_revisits.py --save True 
```

You need to update the revisit file to add sequence 22
```highlight
python save_poses_22.py --save True --data_path **kitti_dir_path**
```


# Installation of LoGG3D-Net

DSI-3D use [LoGG3D-Net](https://github.com/csiro-robotics/LoGG3D-Net/tree/main) so we also need to reproduce their environment.
Since installing torchsparse-1.4.0 can be challenging with admin rights, we relied on an existing Jean Zay module (pytorch-gpu/py3/1.10.1) to pre-compute the LoGG3D-Net features. 
You will also need to download the pretrained LoGG3D-Net model and place it in the checkpoints/ folder.

```highlight
**Load a GPU**
module load pytorch-gpu/py3/1.10.1
```
You first need to generate the positive pairs for each point cloud sequence:
```highlight
cd LoGG3D-Net
python  utils/data_utils/kitti_tuple_mining.py
```
This will create the files positive_sequence_D-3_T-0.json and positive_sequence_D-20_T-0.json in LoGG3D-Net/config/kitti_tuples/.
Copy these files into each sequence folder (this step also simplifies the use of sequence 22).


To generate and save the descriptors, run the following commands:

```highlight
python LoGG3D-Net/evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 0 \
       --checkpoint_name '/checkpoints/kitti_10cm_loo/2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth' \
       --skip_time 30 \
       --save_global_desc True

```

```highlight
python LoGG3D-Net/evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 2 \
       --checkpoint_name '/checkpoints/kitti_10cm_loo/2021-09-14_05-55-20_3n24h_Kitti_v10_q29_10s2_262448.pth' \
       --skip_time 30 \
       --save_global_desc True
```


```highlight
python LoGG3D-Net/evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 5 \
       --checkpoint_name '/checkpoints/kitti_10cm_loo/2021-09-14_06-11-58_3n24h_Kitti_v10_q29_10s5_262449.pth' \
       --skip_time 30 \
       --save_global_desc True    
```

```highlight
python LoGG3D-Net/evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 6 \
       --checkpoint_name '/checkpoints/kitti_10cm_loo/2021-09-14_06-43-47_3n24h_Kitti_v10_q29_10s6_262450.pth' \
       --skip_time 30 \
       --save_global_desc True      
```

```highlight
python LoGG3D-Net/evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 7 \
       --checkpoint_name '/checkpoints/kitti_10cm_loo/2021-09-14_08-34-46_3n24h_Kitti_v10_q29_10s7_262451.pth' \
       --skip_time 30 \
       --save_global_desc True 
```
```highlight
python LoGG3D-Net/evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 8 \
       --checkpoint_name '/checkpoints/kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth' \
       --skip_time 30 \
       --save_global_desc True
```

For the sequence 22, we use the checkpoint 2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth as:
```highlight
python LoGG3D-Net/evaluation/evaluate.py \
       --eval_dataset 'KittiDataset' \
       --kitti_dir **kitti_dir_path** \
       --kitti_eval_seq 22 \
       --checkpoint_name '/checkpoints/kitti_10cm_loo/2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth' \
       --skip_time 30 \
       --save_global_desc True
```


# Dataset indexing

You will need to compute the mapping dictionary between point cloud names and their new indexes. This step only needs to be done once.
```highlight
**Load a GPU**
module load anaconda-py3/2023.03
conda activate DSI_3D
```
## Dataset indexing (Hilbert curve)
To use the Hilbert curve indexation strategy, you need to save one dictionary per sequence.

```highlight
python compute_hilbert_index.py --eval_seq 0 --p 17 --save True --data_path **kitti_dir_path**
```

```highlight
python compute_hilbert_index.py --eval_seq 2 --p 17 --save True --data_path **kitti_dir_path**
```

```highlight
python compute_hilbert_index.py --eval_seq 5 --p 17 --save True --data_path **kitti_dir_path**
```

For sequence 06, our pretrained model was trained using a Hilbert curve at iteration 16.
```highlight
python compute_hilbert_index.py --eval_seq 6 --p 16 --save True --data_path **kitti_dir_path**
```

```highlight
python compute_hilbert_index.py --eval_seq 7 --p 17 --save True --data_path **kitti_dir_path**
```

```highlight
python compute_hilbert_index.py --eval_seq 8 --p 17 --save True --data_path **kitti_dir_path**
```

```highlight
python compute_hilbert_index.py --eval_seq 22 --p 20 --save True --data_path **kitti_dir_path**
```

## Dataset indexing (GPS)
You can compute the GPS indexes with:
```highlight
python -m pdb compute_gps_index.py --eval_seq 0 --gpsround 100 --data_path **kitti_dir_path**
```

```highlight
python -m pdb compute_gps_index.py --eval_seq 2 --gpsround 100 --data_path **kitti_dir_path**
```

```highlight
python -m pdb compute_gps_index.py --eval_seq 5 --gpsround 100 --data_path **kitti_dir_path**
```

```highlight
python -m pdb compute_gps_index.py --eval_seq 6 --gpsround 100 --data_path **kitti_dir_path**
```

```highlight
python -m pdb compute_gps_index.py --eval_seq 7 --gpsround 100 --data_path **kitti_dir_path**
```

```highlight
python -m pdb compute_gps_index.py --eval_seq 8 --gpsround 100 --data_path **kitti_dir_path**
```

```highlight
python -m pdb compute_gps_index.py --eval_seq 22 --gpsround 100 --data_path **kitti_dir_path**
```
## Dataset indexing (hierarchical)
You can compute the hierarchical indexes with:
However, if you want to evaluate our model trained with the hierarchical setup, you will need to use our dictionary file (hierarchical.json).
```highlight
python compute_hierarchical_index.py --eval_seq 0 --data_path **kitti_dir_path** --save True
```
```highlight
python compute_hierarchical_index.py --eval_seq 2 --data_path **kitti_dir_path** --save True
```
```highlight
python compute_hierarchical_index.py --eval_seq 5 --data_path **kitti_dir_path** --save True
```
```highlight
python compute_hierarchical_index.py --eval_seq 6 --data_path **kitti_dir_path** --save True
```
```highlight
python compute_hierarchical_index.py --eval_seq 7 --data_path **kitti_dir_path** --save True
```
```highlight
python compute_hierarchical_index.py --eval_seq 8 --data_path **kitti_dir_path** --save True
```
```highlight
python compute_hierarchical_index.py --eval_seq 22 --data_path **kitti_dir_path** --save True
```

# Training

```highlight
**Load a GPU**
module load anaconda-py3/2023.03
conda activate DSI_3D
```

You may need to change LABEL_MODE to choose an indexation strategy.

```highlight
source train.sh
```
# Checkpoint
coming soon


# Inference

To reproduce the results obtained with our pretrained model, simply run the provided .sh scripts.

```highlight
**Load a GPU**
module load anaconda-py3/2023.03
conda activate DSI_3D
```

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

For Positional Structured identifiers with Hilbert curve indexing on the sequence 00

```highlight
source unit_test_hilbert.sh
```
