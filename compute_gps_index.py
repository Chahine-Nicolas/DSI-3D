import numpy as np
import json
from module_loader_kitti_pose import load_poses_from_txt

eval_dataset = 'Kitti'

list_all_pose = []

eval_seq = 5

WORK_PATH = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el"
kitti_dir = WORK_PATH+"/datas/datasets/"


eval_seq = '%02d' % eval_seq
sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
tfs, pose = load_poses_from_txt(sequence_path + 'poses.txt')

min_bbox = np.min(pose,0) 
pose = pose - min_bbox

for query in range(len(pose)):
    list_all_pose.append(pose[query])

gpsround = 100
def label2gps(label_id, positions_database) :
    label_id_gps = positions_database[int(label_id)]
    xx = round(label_id_gps[0]*gpsround)
    yy = round(label_id_gps[1]*gpsround)
    xx_str = f'{xx:05}'
    yy_str = f'{yy:05}'
    res_str = ''.join(x + y for x, y in zip(xx_str, yy_str))
    res_str += xx_str[len(yy_str):] + yy_str[len(xx_str):]
    return res_str

dictio = {}
for i in range(len(list_all_pose)):
    print( '%06d' % i, label2gps(i, list_all_pose))
    if label2gps(i, list_all_pose) in dictio:
        continue
    dictio[label2gps(i, list_all_pose)] = '%06d' % i

save = False
if save:
    with open(sequence_path + "gps.json", 'w', encoding ='utf8') as json_file: 
            json.dump(dictio, json_file, allow_nan=False) 
    print("saved dictio", sequence_path + "gps.json")