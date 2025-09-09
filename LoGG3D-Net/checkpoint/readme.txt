Data related to the ICRA 2022 paper: LoGG3D-Net: Locally Guided Global Descriptor Learning for 3D Place Recognition

This folder contains the checkpoints used for creating the results in Table 2 of the paper. 
- kitti_10cm_loo contains 6 checkpoints corresponding to the leave-one-out (loo) training. Trained on the first 11 sequences to evaluate on the 6 sequences with revisits (00, 02, 05, 06, 07 and 08).
-- In the checkpoint names, 10sx means that it was trained on 10 sequences while leaving out sequence x. 
-- eg. checkpoint '2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447' was trained by leaving out sequence 0. So this is the checkpoint that should be used for evaluating sequence 0. 
- mulran_10cm contains 1 checkpoint trained on sequences DCC1, DCC2, Riverside1 and Riverside3

Contact: kavisha.vidanapathirana@data61.csiro.au