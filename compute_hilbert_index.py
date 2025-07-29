import os
import matplotlib.pyplot as plt 
from module_loader_kitti_pose import * # add for more metrics
from hilbertcurve.hilbertcurve import HilbertCurve
import json
##################################
# read pos
##################################
eval_dataset = 'Kitti'
eval_seq = 6
kitti_dir = os.getenv('WORKSF') + "/datas/datasets/"
eval_seq = '%02d' % eval_seq
sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
tfs, pose = load_poses_from_txt(sequence_path + 'poses.txt')

min_bbox = np.min(pose,0) 
pose = pose - min_bbox


p = 16 
n = 2   

hilbert_curve = HilbertCurve(p, n)

dict_hibert = {}
for i in range(len(pose)):

    label_id_gps = pose[int(i)]
    xx = round(label_id_gps[0]*100)
    yy = round(label_id_gps[1]*100)
    
    res_str = str(hilbert_curve.distances_from_points([[xx, yy]])[0])

    print(i, str( res_str )  )
    dict_hibert[ str( res_str ) ] = '%06d' % i



json_path = sequence_path + 'hilbert_16.json'

save = False
if save:
    with open(json_path, "w") as json_file:
        json.dump(dict_hibert, json_file)  
    print("docid_map saved at ", json_path)


