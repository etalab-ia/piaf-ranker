import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import CamembertTokenizer,CamembertModel



    
    
class Full_Network(nn.Module):

    def __init__(self):

        super(Full_Network, self).__init__()
        
        self.camembert = CamembertModel.from_pretrained("camembert-base")
        self.fc1 = nn.Linear(768, 768)
        self.relu =  nn.ReLU()
        self.fc2 = nn.Linear(768,1)
        self.dropout = nn.Dropout(0.30)

    def forward(self, token,mask):
        x = self.camembert(token,mask)['pooler_output']
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
                
        return x
    
