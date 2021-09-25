import pandas as pd
import re


import torch 
import json
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from model import Network,Full_Network
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from config.test_config import parameters
from utils.utils import tokenize,evaluate


k = parameters['k']
model = parameters['model']
complementary = parameters['complementary']
full = parameters["full"]
alpha = parameters["alpha"]
beta = parameters["beta"]


order_by = 'MonoBERT_Score' if complementary==False else 'complementary'



def load_raw_data(path = '../data/raw/MonoBERT_test.json'):
    with open(path,encoding="utf-8") as json_file:
                data = json.load(json_file)
    
    
    #remove question with no_retrieved docs
    for id_,data_ in enumerate(data):
        if len(data_['retrieved_contexts']) == 0:
            data.pop(id_)
            
    return data

def select_docs(json_data):

        data_list = []
        id_ = 0
        
        for query,datum in enumerate(json_data):
            
            for doc_id,text in enumerate(datum['retrieved_contexts']):
                data_list.append([text['id'],
                                  datum['question'],
                                  text['context'],
                                  ])

        return data_list

    
def generate_queries(data):

        queries= []

        for entry in data:
            query,doc = entry[1],entry[2]
            input_ = ' '.join([query,'</s>',doc])
            queries.append(input_)            

        queries = tokenize(queries)

        return queries 

    
def run_model(tokens,masks , train_on_gpu = False , model_weights ='../models/model_train_full_finetune.pt' ):
    
    train_on_gpu = True
    
    model = Full_Network()    
    if train_on_gpu :
        print('Trainning on GPU')
        model = model.cuda()

    model.load_state_dict(torch.load(model_weights))

    tk0 = tqdm(zip(tokens,masks), total=int(len(masks)))

    results=[]
    embdings = []
    # iterate over test data
    for batch_idx, (token,mask) in enumerate(tk0):
        model.eval()
        if train_on_gpu:
            token  = token.cuda()
            mask = mask.cuda()

        with torch.no_grad() :
            output = model(token.view(1,-1),mask.view(1,-1))

        results.append(output.cpu())
        
    return results


    


def order_data(data , results , order_by = order_by):
    counter_results = 0 
    for query_number,query in enumerate(data):
        for doc_number, doc in enumerate(query['retrieved_contexts']):
            score = float(results[counter_results])  
            min_ = data[query_number]['retrieved_contexts'][-1]['score']
            max_ = data[query_number]['retrieved_contexts'][0]['score']
            BM25_score = data[query_number]['retrieved_contexts'][doc_number]['score']
            data[query_number]['retrieved_contexts'][doc_number]['MonoBERT_Score'] = score
            data[query_number]['retrieved_contexts'][doc_number]['normalized_BM25'] = normalized_BM25 =(BM25_score - min_ + 1)/(max_ - min_ + 1)
            #data_test[query]['negative_contexts'][doc]['final'] = score + 27*(BM25_score/score_agg)
            data[query_number]['retrieved_contexts'][doc_number]['complementary'] = alpha*score + beta*normalized_BM25
            counter_results += 1
            
    for datum in data:
        datum['retrieved_contexts'] = sorted(datum['retrieved_contexts'], key=lambda k: k[order_by],reverse=True) 
        
    return data

    
    
    

if __name__== "__main__":
    
    data = load_raw_data('../data/raw/ranker_test.json')
    data_list = select_docs(data)
    queries = generate_queries(data_list)
    results = run_model(queries['input_ids'],queries['attention_mask'])
    ordered_data = order_data(data,results)
    metrics = evaluate(ordered_data,k)
    print(metrics)
    
    
    
    
    