import numpy as np
import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

list_seq = [0, 2, 5, 6, 7, 8] 
#list_seq = [6] 
root_path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/"


class SequenceDataset(Dataset):
    def __init__(self, list_seq, data, root_path):
        self.samples = []
        self.labels = []
        self.seq_to_idx = {seq: i for i, seq in enumerate(list_seq)}  # converti 0,2,5 en 0,1,2..

        for seq in list_seq:
            seq_str = f"{seq:02d}"
            sequence_path = os.path.join(root_path, seq_str, "logg_desc")

            for j in range(len(data[seq_str])):
                file_path = os.path.join(sequence_path, f"{j:06d}.pt")

                if j % 5 == 0:
                    continue
                
                if os.path.exists(file_path):
                    vec = torch.load(file_path).to(torch.float32) 
                    self.samples.append(vec)
                    self.labels.append(self.seq_to_idx[seq])
                       

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

"""
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)  # Output: one neuron per expert
        )

    def forward(self, x):
        return self.model(x)
"""

# 22
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout 30%

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

    def forward(self, x):
        return self.model(x)


def predict_expert(model, feature_vector, device):
    with torch.no_grad():
        feature_vector = feature_vector.to(device).unsqueeze(0)  # Add batch dimension
        output = model(feature_vector)
        predicted_expert_idx = torch.argmax(output).item()

        # proba
        m = nn.Softmax(dim=1)
        prob_seq = m(output)

    return list_seq[predicted_expert_idx], output[0][predicted_expert_idx], prob_seq[0][predicted_expert_idx] 



def main():

    with open("/lustre/fswork/projects/rech/dki/ujo91el/code/these_place_reco/LoGG3D-Net/config/kitti_tuples/is_revisit_D-3_T-30.json") as f:
        data = json.load(f)
    root_path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/"
    print("Load dataset")
    dataset = SequenceDataset(list_seq, data, root_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device="cuda"
    # load gate

    #from train_relu import ExpertClassifier

    # Define the model with the same architecture
    gate_model = ExpertClassifier()
    
    # Load the saved weights
    print("load ","expert_router.pth")
    gate_model.load_state_dict(torch.load("expert_router_22.pth"))
    gate_model.to(device="cuda")
    gate_model.eval()  # Set the model to evaluation mode
    

    """
    #import pdb; pdb.set_trace()
    
    seq_str =  cfg['DATA_CONFIG']['SEQ']
    root_path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/"
    desc_path = os.path.join(root_path, seq_str, "logg_desc")
    file_path = os.path.join(desc_path, f"{query_idx:06d}.pt")
    
    test_feature = torch.load(file_path).to(torch.float32)

    best_expert, expert_seq, score, prob = predict_expert(gate_model, test_feature, 'cuda')
    print(file_path)
    print(f"Predicted expert: {best_expert}, Expected expert: {int(seq_str)}, Score: {score}, proba: {prob} ")
    """

    # evaluation
    print("start evaluation")
    seq_to_idx = {seq: i for i, seq in enumerate(list_seq)}
    hit, num = 0, 0
    seen_proba = []

    # matrice de confusion
    y_true = []
    y_pred = []
    
    for seq in list_seq:
            seq_str = f"{seq:02d}"
            sequence_path = os.path.join(root_path, seq_str, "logg_desc")
            
            for j in range(len(data[seq_str])):
                file_path = os.path.join(sequence_path, f"{j:06d}.pt")
    
                #if j % 5 != 0:
                    #continue
                
                
                #test_feature_0 = torch.load(file_path).to(torch.float32)  # Force float32 

                
                desc_path = os.path.join(root_path, "22", "logg_desc")
                file_path = os.path.join(desc_path, f"{num:06d}.pt")
                test_feature = torch.load(file_path).to(torch.float32)

    
                num +=1

                
                
                #best_expert0, score0, prob0 = predict_expert(gate_model, test_feature_0, device)
                best_expert, score, prob = predict_expert(gate_model, test_feature, device)

                y_true.append(int(seq_str))
                y_pred.append(best_expert)
                
                print(file_path)
                #print(f"Predicted expert: {best_expert0}, Expected expert: {int(seq_str) }, Score: {score0}, proba: {prob0} ")
                print(f"Predicted expert: {best_expert}, Expected expert: {int(seq_str) }, Score: {score}, proba: {prob} ")
                seen_proba.append(prob.cpu().numpy())
                
                if best_expert ==  int(seq_str):
                    hit += 1
                    
    print("correct prediction (%): ", hit / num)
    print("average proba: ", np.mean(seen_proba) )

    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy

    conf_mat = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=numpy.array([0, 2, 5, 6, 7, 8]))

    disp.plot()
    plt.savefig("test.jpg")
    import pdb; pdb.set_trace()
    
if __name__ == '__main__':
    main()
