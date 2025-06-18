from models.pipeline_factory import get_pipeline
from models.pipelines.pipeline_utils import make_sparse_tensor
import numpy as np
import random
import torch

def random_rotate(xyzr, r_angle=360, is_random=True, add_noise=True, rand_tr=False):
    # If is_random = True: Rotate about z-axis by random angle upto 'r_angle'.
    # Else: Rotate about z-axis by fixed angle 'r_angle'.
    r_angle = (np.pi/180) * r_angle
    if is_random:
        r_angle = r_angle*np.random.uniform()
    cos_angle = np.cos(r_angle)
    sin_angle = np.sin(r_angle)
    rot_matrix = np.array([[cos_angle, -sin_angle, 0],
                        [sin_angle, cos_angle, 0],
                        [0,         	0,  	1]])
    scan = xyzr[:, :3]
    int = xyzr[:, 3].reshape((-1, 1))
    augmented_scan = np.dot(scan, rot_matrix)
    
    if add_noise:
        n_sigma = 0.01  # Add gaussian noise
        noise = np.clip(n_sigma * np.random.randn(*
                        augmented_scan.shape), -0.03, 0.03)
        augmented_scan = augmented_scan + noise
    
    if rand_tr:
        tr_xy_max, tr_z_max = 1.5, 0.25
        tr_xy = np.clip(np.random.randn(1, 2), -tr_xy_max, tr_xy_max)
        tr_z = np.clip(0.1*np.random.randn(1, 1), -tr_z_max, tr_z_max)
        tr = np.hstack((tr_xy, tr_z))
        augmented_scan = augmented_scan + tr
    
    augmented_scan = np.hstack((augmented_scan, int))
    return augmented_scan.astype(np.float32)
    
def occlude_scan(scan, angle=30):
    # Remove points within a sector of fixed angle (degrees) and random heading direction.
    thetas = (180/np.pi) * np.arctan2(scan[:, 1], scan[:, 0])
    heading = (180-angle/2)*np.random.uniform(-1, 1)
    occ_scan = np.vstack(
        (scan[thetas < (heading - angle/2)], scan[thetas > (heading + angle/2)]))
    return occ_scan.astype(np.float32)


def get_logg3d_net_desc(eval_seq, lidar_values, input_mod, voxel_size=0.1, random_rotation = False, random_occlusion = False, random_scale = False):

    model = get_pipeline('LOGG3D')
    
    if eval_seq == '00' or eval_seq == '22' :
        save_path =  "/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth"
    elif eval_seq == '02':
        save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_05-55-20_3n24h_Kitti_v10_q29_10s2_262448.pth'
    elif eval_seq == '05':
        save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_06-11-58_3n24h_Kitti_v10_q29_10s5_262449.pth'
    elif eval_seq == '06':
        save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_06-43-47_3n24h_Kitti_v10_q29_10s6_262450.pth'
    elif eval_seq == '07':
        save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_08-34-46_3n24h_Kitti_v10_q29_10s7_262451.pth'
    elif eval_seq == '08':
        save_path =  '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth'

    checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model = model.cuda()
    model.eval()

    lidar_file = '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/'+eval_seq+'/velodyne/' + lidar_values[input_mod][0][1]
    lidar_pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    
    print(len(lidar_pc))
    print(len(lidar_values['points']))
    lidar_pc = lidar_values['points'].cpu().numpy()
    
    max_scale = 1.2
    min_scale = 0.8

    lidar_pc2 = lidar_pc
    if random_rotation:
        lidar_pc2 = random_rotate(lidar_pc2)
    if random_occlusion:
        lidar_pc2 = occlude_scan(lidar_pc2)
    if random_scale and random.random() < 0.95:
        scale = min_scale + \
            (max_scale - min_scale) * random.random()
        lidar_pc2 = scale * lidar_pc2
    
    input2 = make_sparse_tensor(lidar_pc2, voxel_size).cuda()
    output_desc2, output_feats2 = model(input2)  # .squeeze()
    #output_feats2 = output_feats2[0]
    #global_descriptor2 = output_desc2.cpu().detach().numpy(
    return output_desc2