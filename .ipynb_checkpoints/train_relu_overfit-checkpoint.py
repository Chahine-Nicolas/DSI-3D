import numpy as np
import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

list_seq = [0, 2, 5, 6, 7, 8] 
length_seq = [4541, 4661, 2701, 1101, 1101, 4071]
root_path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/"
    

class multiSequenceDataset(Dataset):
    def __init__(self, list_seq, data, root_path):
        self.samples = []
        self.labels = []
        self.seq_to_idx = {seq: i for i, seq in enumerate(list_seq)}  # converti 0,2,5 en 0,1,2..

        for seq in list_seq:
            seq_str = f"{seq:02d}"
            sequence_path = os.path.join(root_path, seq_str, "logg_desc")

            for j in range(len(data[seq_str])):
                file_path = os.path.join(sequence_path, f"{j:06d}.pt")
                
                if os.path.exists(file_path):
                    vec = torch.load(file_path).to(torch.float32) 
                    self.samples.append(vec)
                    self.labels.append(self.seq_to_idx[seq])
                       

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]



class SequenceDataset(Dataset):
    def __init__(self, root_path, length_seq):
        self.samples = []
        self.labels = []
        self.length_seq = length_seq  # Lengths of each sequence
        self.cumulative_lengths = self.compute_cumulative_lengths(length_seq)

        # Load data from sequence folder "22"
        sequence_path = os.path.join(root_path, "22", "logg_desc")

        total_samples = sum(length_seq)
        for j in range(total_samples):
            file_path = os.path.join(sequence_path, f"{j:06d}.pt")

            #if j % 5 == 0:  # Skip every 5th sample
                #continue

            if os.path.exists(file_path):
                vec = torch.load(file_path).to(torch.float32)
                self.samples.append(vec)
                self.labels.append(self.get_label_from_index(j))
                
    def compute_cumulative_lengths(self, length_seq):
        """Computes cumulative sequence lengths for mapping indices to labels."""
        cumulative = [0]  # Start from 0
        for length in length_seq:
            cumulative.append(cumulative[-1] + length)
        return cumulative

    def get_label_from_index(self, index):
        """Finds the corresponding sequence label based on index."""
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= index < self.cumulative_lengths[i + 1]:
                return i  # Label is the index in `list_seq`
        return -1  # Should never happen if data is correct

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]




"""
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
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
"""
class ExpertClassifier(nn.Module):
    def __init__(self, input_dim=256, num_experts=len(list_seq)):
        super(ExpertClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Normalize activations

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),  # Prevent overfitting

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_experts)  # Output logits for experts
        )

    def forward(self, x):
        return self.model(x)
"""
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

    device = 'cuda'
    print("Load dataset")
    dataset =  SequenceDataset(root_path, length_seq)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    dataset =  SequenceDataset(root_path, length_seq)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #import pdb; pdb.set_trace()
    
    """
    class ExpertClassifier(nn.Module):
        def __init__(self, input_dim=256, num_experts=len(list_seq)):
            super(ExpertClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, num_experts)  # Output: one neuron per expert
            )
    
        def forward(self, x):
            return self.model(x) # 0.9668
    """  
    
    """ 
    class ExpertClassifier(nn.Module):
        def __init__(self, input_dim=256, num_experts=len(list_seq)):
            super(ExpertClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_experts)  # Output: one neuron per expert
            )
    
        def forward(self, x):
            return self.model(x)
    """
    
    # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

    print("Initialize model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpertClassifier().to(device)
    
    print("model", model)
    
    print("Define loss and optimizer")
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print("Training loop")
    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
    
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
    
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
    
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Track performance
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
    
    # Save trained model
    torch.save(model.state_dict(), "expert_router_overfit.pth")
    import pdb; pdb.set_trace()
    model.eval()
    
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
    
                num +=1 
                
                test_feature = torch.load(file_path).to(torch.float32)  # Force float32 
    
                best_expert, score, prob = predict_expert(model, test_feature, device)
                print(file_path)
                print(f"Predicted expert: {best_expert}, Expected expert: {int(seq_str) }, Score: {score}, proba: {prob} ")
                seen_proba.append(prob.cpu().numpy())
                
                y_true.append(int(seq_str))
                y_pred.append(best_expert)
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
    plt.savefig("train_over.jpg")
    import pdb; pdb.set_trace()
            

if __name__ == '__main__':
    main()