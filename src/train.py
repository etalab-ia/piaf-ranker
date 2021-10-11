import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertTokenizer,CamembertModel



from transformers import AdamW

import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sklearn.utils import class_weight
from sklearn.preprocessing import normalize


from model import Network,Full_Network

from utils.utils import generate_masks_and_tokens
from config.train_config import parameters



#load parameters
train_data = Path(parameters["train_data"])
dev_data = Path(parameters["dev_data"])
batch_size = parameters["batch_size"]
epochs = parameters["epochs"]
lr = parameters["lr"]  
eps = parameters["eps"] 
margin = parameters["margin"]
IDLE_EPOCHS_STOP = 5

#load data and prepare datasets
print('Loading training data ...')
train_data = generate_masks_and_tokens(train_data)
print('Loading dev data ...')
dev_data = generate_masks_and_tokens(dev_data)



#set up the dataloaders
train_dataset = TensorDataset(train_data['positive_tokens'],
                              train_data['positive_masks'],
                              train_data['negative_tokens'],
                              train_data['negative_masks'],
                              )

train_dataloader = DataLoader(train_dataset,
                              sampler = RandomSampler(train_dataset),
                              batch_size = batch_size)





dev_dataset = TensorDataset(dev_data['positive_tokens'],
                            dev_data['positive_masks'],
                            dev_data['negative_tokens'],
                            dev_data['negative_masks'],
                            )

dev_dataloader = DataLoader(dev_dataset,
                              sampler = RandomSampler(dev_dataset),
                              batch_size = batch_size)




#load the model
model = Full_Network()


train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(train_on_gpu):
    print('Training on GPU ')
else: 
    print('Training on CPU ')
    

if train_on_gpu:
        model.cuda()

        
        
#set the optimizer
optimizer = AdamW(model.parameters(),
                  lr = lr, 
                  eps = eps)

#set the loss
criterion = nn.MarginRankingLoss(margin=margin)




valid_loss_min = np.Inf 
idle_counter = 0
train_loss_history = []   #average train loss per epoch list
dev_loss_history = []     #average dev loss per epoch list
train_time_history = []   #total train time per epoch list
dev_time_history = []     #total dev time per epoch list




for epoch in range(0, epochs):
     
        
    tqdm.write("")
    tqdm.write(f'########## Epoch {epoch+1} / {epochs} ##########')
    tqdm.write('Training...')
    
    
    
    
    ###################
    # train the model #
    ###################
    start = time.time()

    train_loss = 0.0
    model.train()
    
    tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
    
    for batch_idx, (pos_token,pos_mask,neg_token,neg_mask) in enumerate(tk0):
        
            
        optimizer.zero_grad()
        targets = torch.ones(pos_token.shape[0], 1)
        
        if train_on_gpu:
            pos_token,pos_mask = pos_token.cuda() , pos_mask.cuda()
            neg_token,neg_mask = neg_token.cuda() , neg_mask.cuda()
            targets = targets.cuda()
    
        
        pred_pos = model(pos_token,pos_mask)
        pred_neg = model(neg_token,neg_mask)

        if train_on_gpu:
            targets = targets.cuda()
        loss = criterion(pred_pos, pred_neg,targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*pos_token.size(0)
        
    end = time.time()
    train_time = end - start 
    ######################    
    # validate the model #
    ######################
    start = time.time()
    valid_loss = 0.0
    
    tk1 = tqdm(dev_dataloader, total=int(len(dev_dataloader)))
    model.eval()
    for batch_idx, (pos_token,pos_mask,neg_token,neg_mask) in enumerate(tk1):
        
            
        targets = torch.ones(pos_token.shape[0], 1)
        
        if train_on_gpu:
            pos_token,pos_mask = pos_token.cuda() , pos_mask.cuda()
            neg_token,neg_mask = neg_token.cuda() , neg_mask.cuda()
            targets = targets.cuda()
    
        
        pred_pos = model(pos_token,pos_mask)
        pred_neg = model(neg_token,neg_mask)
        
        loss = criterion(pred_pos, pred_neg,targets)
        valid_loss += loss.item()*pos_token.size(0)
        
    end = time.time()
    dev_time = end - start 
    
    
    train_loss = train_loss/len(train_dataloader.sampler)
    valid_loss = valid_loss/len(dev_dataloader.sampler)
    
    train_loss_history.append(train_loss)
    dev_loss_history.append(valid_loss)
    train_time_history.append(train_time)
    dev_time_history.append(dev_time)
    
     
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    #if validation loss decreased save the model and reset the counter
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), '../models/model_train_full_finetune.pt')
        valid_loss_min = valid_loss
        idle_counter = 0
    
    #increment the counter if validation did not decrease
    else:
        idle_counter +=1
     
    #save train stats
    train_stats = {'train_loss' : train_loss_history,
                   'dev_loss' : dev_loss_history,
                   'train_time' : train_time_history,
                   'dev_time'   : dev_time_history}

    with open('../results/results_full.pickle', 'wb') as handle:
        pickle.dump(train_stats, 
                    handle)
        
    #check if the counter is equal to the limit (early stopping) 
    if idle_counter == IDLE_EPOCHS_STOP:
        print('end of trainning at epoch',epoch)
        break
 

    
    
