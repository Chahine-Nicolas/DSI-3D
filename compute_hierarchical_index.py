import os
import sys
import glob
import random
import numpy as np
import logging
import json
import torch
import math
import argparse
from sklearn.cluster import KMeans

def elementwise_str_concat(prefix_list, suffix_list):
    return [str(prefix) + str(suffix) for prefix, suffix in zip(prefix_list, suffix_list)]

def reorder_to_original(J, X, clusters):
    # Créer une liste d'indices pour trier `J` selon l'ordre original des documents
    original_order = np.concatenate(clusters).argsort()
    # Réordonner `J` en utilisant cet ordre original
    return [J[i] for i in original_order]

# verification duplicate
def allDifferent1D(lst):
    return len(lst) == len(set(lst))

def generate_semantic_ids(X, c=10):
    # Effectuer le clustering des documents en k=10 clusters et obtenir les labels directement
    k = 10
    
    labels = KMeans(n_clusters=k).fit_predict(X)
    
    # Grouper les indices des documents par label
    clusters = [np.where(labels == i)[0] for i in range(k)]
    
    J = []
    for i in range(k):
        current_cluster_size = len(clusters[i])
        Jcurrent = [str(i)] * current_cluster_size
        
        if current_cluster_size > c:
            # Appliquer la fonction récursive à ce cluster
            sub_cluster_embeddings = X[clusters[i]] # sélectionne les descripteurs du cluster courant
            Jrest = generate_semantic_ids(sub_cluster_embeddings, c) # génère les noms, récursif
            
        else:
            # Générer des indices numériques pour les documents dans ce cluster
            Jrest = list(map(str, range(current_cluster_size)))
        
        # Concaténer les préfixes et les suffixes pour obtenir les identifiants du cluster
        Jcluster = elementwise_str_concat(Jcurrent, Jrest)
        J.extend(Jcluster)
    
    # Réordonner les identifiants pour correspondre à l'ordre original des documents
    J = reorder_to_original(J, X, clusters) # réordonne les ids hierarichique selon l'ordre des descripteurs (et non des clusters)

    # verification duplicate
    def allDifferent1D(lst):
        return len(lst) == len(set(lst))
    if allDifferent1D(J) == False:
        print("Il y a des doublons !")

    return J

def compute_hierarchical_clustering(eval_seq, num_queries, data_path, save):
    log3dnet_dir=os.getenv('LOG3DNET_DIR')
    ## ==== Kitti =====
    print("kitti dataset")
   
    eval_seq = '%02d' % eval_seq
    sequence_path = data_path + 'sequences/' + eval_seq + '/'
     
    embeddings = []
    for query_idx in range(num_queries):

        #ids = input_data['ids'][0]
        padded_string = str(query_idx).zfill(6)
        print(padded_string)
        #import pdb; pdb.set_trace()                
        log_desc = sequence_path + "/logg_desc/" + padded_string + '.pt'
    
        xx = torch.load(log_desc)
        #xx1 = torch.load(fname)
        embeddings.append(xx)

    emb_cpu = torch.stack(embeddings).detach().cpu().numpy()
    J = generate_semantic_ids(emb_cpu , c=10)
    docid_map = {k: J[k] for k in range(len(J)) } # creat dict
    for i in range(len(J)):
        print(f"Document {i}: Identifier {J[i]}")

    json_path = sequence_path + '/hierarchical.json'
    # # Print the generated identifiers
    for doc_idx, doc_id in docid_map.items():
        print(f"Document {doc_idx}: Identifier {doc_id}")

    if save:
        with open(json_path, "w") as json_file   :
            json.dump(docid_map, json_file)  
        print("docid_map saved at ", json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical mapping for KITTI poses")
    parser.add_argument("--eval_seq", type=int, required=True, help="Sequence number to evaluate (e.g., 6)")
    parser.add_argument("--num_queries", type=int, default=4541, help="number of queries")
    kitti_dir = os.getenv("WORKSF") + "/datas/datasets/"
    parser.add_argument("--data_path", type=str,  default=kitti_dir, required=True, help="Dataset path")
    parser.add_argument("--save", type=bool, default=False, help="Save result as JSON")
    
    args = parser.parse_args()
    compute_hierarchical_clustering(args.eval_seq, args.num_queries, args.data_path, args.save) 

# python -m pdb  compute_hierarchical_index.py --eval_seq 0 --num_queries 4541 --data_path '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/'



