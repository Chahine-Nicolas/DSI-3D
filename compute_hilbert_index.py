import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from module_loader_kitti_pose import load_poses_from_txt
from hilbertcurve.hilbertcurve import HilbertCurve

def main(eval_seq: int, p: int, save: bool, data_path: str):
    ##################################
    # Setup paths
    ##################################
    eval_seq_str = f"{eval_seq:02d}"
    sequence_path = os.path.join(data_path, "sequences", eval_seq_str)

    ##################################
    # Load poses
    ##################################
    tfs, pose = load_poses_from_txt(os.path.join(sequence_path, "poses.txt"))

    # Normalize pose so that min is (0,0,...)
    min_bbox = np.min(pose, axis=0)
    pose = pose - min_bbox

    ##################################
    # Hilbert curve mapping
    ##################################
    n = 2  # dimension fixed to 2
    hilbert_curve = HilbertCurve(p, n)

    dict_hilbert = {}
    for i, coords in enumerate(pose):
        xx = round(coords[0] * 100)
        yy = round(coords[1] * 100)

        res_str = str(hilbert_curve.distances_from_points([[xx, yy]])[0])
        print(i, res_str)
        dict_hilbert[res_str] = f"{i:06d}"

    ##################################
    # Save JSON (if requested)
    ##################################

    if save:
        json_path = os.path.join(sequence_path, f"hilbert.json")
        with open(json_path, "w") as json_file:
            json.dump(dict_hilbert, json_file)
        print("docid_map saved at", json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hilbert curve mapping for KITTI poses")
    kitti_dir = os.getenv("WORKSF") + "/datas/datasets/"
    parser.add_argument("--data_path", type=str,  default=kitti_dir, required=True, help="Dataset path")
    parser.add_argument("--eval_seq", type=int, required=True, help="Sequence number to evaluate (e.g., 6)")
    parser.add_argument("--p", type=int, default=16, help="Hilbert curve iterations (default: 16)")
    parser.add_argument("--save", type=bool, default=False, help="Save result as JSON")

    args = parser.parse_args()
    main(args.eval_seq, args.p, args.save, args.data_path) 

