import os
import sys
import glob
import random
import numpy as np
import logging
import json
import torch
import math
import numpy as np
from sklearn.cluster import KMeans

def hierarchical_clustering(embeddings, c=100, cluster_count=10):
    """
    Perform hierarchical clustering on document embeddings and assign document IDs.

    Args:
        embeddings (np.ndarray): 2D array of document embeddings (num_docs x embedding_dim).
        c (int): Maximum number of documents in a cluster before further clustering is applied.
        cluster_count (int): Number of clusters to create at each level (default 10 for decimal tree).

    Returns:
        dict: A mapping from document index to their assigned identifier.
    """
    def cluster_recursive(embeddings, prefix=''):
        num_docs = embeddings.shape[0]
        
        # If the number of documents is less than or equal to c, assign an arbitrary number.
        if num_docs <= c:
            return {i: prefix + str(i) for i in range(num_docs)}
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=min(cluster_count, num_docs))
        clusters = kmeans.fit_predict(embeddings)
        
        # Dictionary to store the final docid for each document
        docid_map = {}
        
        # Process each cluster recursively
        for cluster_id in range(min(cluster_count, num_docs)):
            # Get the embeddings belonging to the current cluster
            cluster_embeddings = embeddings[clusters == cluster_id]
            
            # Recursively cluster within each sub-cluster and append to the prefix
            cluster_docid_map = cluster_recursive(cluster_embeddings, prefix=prefix + str(cluster_id))
            # Map document IDs
            docid_map.update({doc_idx: cluster_docid_map[i] for i, doc_idx in enumerate(np.where(clusters == cluster_id)[0])})
        
        return docid_map
    
    # Start the recursive clustering from the top level
    return cluster_recursive(embeddings)


def compute_hierarchical_clustering(eval_subset,eval_set,data_collator,tokenizer,cfg):
    eval_seq = 0
    log3dnet_dir=os.getenv('LOG3DNET_DIR')
    ## ==== Kitti =====
    print("kitti dataset")
    kitti_dir= os.getenv('WORKSF') + '/datas/datasets/'
    eval_seq = '%02d' % eval_seq
    sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
    num_queries =  len(eval_subset)
    embeddings = []
    for query_idx in range(num_queries):
        #input_data = data_collator(torch.utils.data.Subset(eval_subset,range(query_idx, query_idx+1)))
        #ids = input_data['ids'][0]
        padded_string = str(query_idx).zfill(6)
        print(padded_string)
        #import pdb; pdb.set_trace()                
        log_desc = sequence_path + "/logg_desc/" + padded_string + '.pt'

        xx = torch.load(log_desc)
        #xx1 = torch.load(fname)
        embeddings.append(xx)


    num_docs = num_queries  # Example number of documents
    embedding_dim = 256  # Example embedding size (BERT embeddings)
    emb_cpu = torch.stack(embeddings).detach().cpu().numpy()
    docid_map = hierarchical_clustering(emb_cpu, c=100, cluster_count=10)
    docid_map_pad = {int(k): v.zfill(6) for k, v in docid_map.items()}
    json_path = sequence_path + '/hierarchical.json'
    # # Print the generated identifiers
    for doc_idx, doc_id in docid_map.items():
        print(f"Document {doc_idx}: Identifier {doc_id}")
    with open(json_path, "w") as json_file   :
        json.dump(docid_map_pad, json_file)        
    #import pdb; pdb.set_trace()                



    
