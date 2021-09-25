import pickle
import random
import json
from tqdm import tqdm
from transformers import CamembertTokenizer
from torch.utils.data import TensorDataset,DataLoader,RandomSampler, SequentialSampler
import torch
import pandas as pd



def initiliaze_tokenizer():
    """
    Initialize CamembertTokenizer 
    output : CamembertTokenizer object
    """
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base',
                                                       sep_token='</s>', 
                                                       cls_token='<s>', 
                                                       do_lower_case=True
                                                        )
    
    return tokenizer



def select_docs(json_data):
        """
        Turn Json to list of documents 
            - deletes positive contexts from retreived
            - add target or class to each context
            
        input :  ['question','positiontexts'=[],retrieved_contexts:[]]
        output : [id,question,context,target]
        """
        data_list = []
        id_ = 0
        
        for query,datum in enumerate(json_data):
            gold_ids = [pos_cntxt['id'] for pos_cntxt in datum['positive_contexts']]
            for doc_id,text in enumerate(datum['positive_contexts']):
                if len(datum['question'].split()) + len(text['context'].split()) <= 509 :
                    data_list.append([text['id'],
                                      datum['question'],
                                      text['context'],
                                      1])

            
            for doc_id,text in enumerate(datum['retrieved_contexts']):
                target = 2 if text['id'] in gold_ids else 0
                if len(datum['question'].split()) + len(text['context'].split()) <= 509 and text['id'] and text['id'] not in gold_ids :
                    data_list.append([text['id'],
                                      datum['question'],
                                      text['context'],
                                      target])

        return data_list



def tokenize(queries):
        """
        Tokenize a list of quesries
        
        input :  [question [SEP] context]
        output : {'input_ids':[],'attention_mask'=[]}
        
        """
    
        tokenizer = initiliaze_tokenizer()

        encoded_queries = tokenizer.batch_encode_plus(queries,
                                                      add_special_tokens=True,
                                                      max_length=512,
                                                      padding='max_length',
                                                      truncation=True, 
                                                      return_attention_mask=True,
                                                      return_tensors="pt"
                                                      )

        return encoded_queries

    
    
def generate_queries_and_labels(data):
    
        """
        Generates queries (tokens and masks) and labels
        
        input :  [id,question,context,target]
        output : ({'input_ids':[],attention_masks=[]},labels)
        
        """
    

        queries,labels= [],[]

        for entry in data:
            query,doc,label = entry[1],entry[2],entry[3]
            input_ = ' '.join([query,'</s>',doc])
            queries.append(input_)
            labels.append(torch.tensor(label))
            

        queries = tokenize(queries)

        return queries,labels 
   
    
def create_pairs(embds,labels):
    """
        Generates positive-negative pairs 
        
        input : 
            - embds : list of tensors
            - labels : list of tensors 
                
        output :
            - positive_cntxts : list of positive tensors
            - negative_cntxts : list of negative tensors
        
    """
    
    positive_cntxts = []
    negative_cntxts = []
    for id_,label in enumerate(labels):
        current_label = int(label)
        if current_label == 1 :
            pos = embds[id_]
        else:
            neg = embds[id_]
            negative_cntxts.append(neg)
            positive_cntxts.append(pos)   
            
    return positive_cntxts,negative_cntxts



def generate_masks_and_tokens(path = './data/raw/train.json'):
    
    
    # load the data
    with open(path,encoding="utf-8") as json_file:
                doc_list = json.load(json_file)
    
    # remove empty queries
    for id_,data_ in enumerate(doc_list):
        if len(data_['retrieved_contexts']) == 0:
            doc_list.pop(id_)
            
    doc_list = select_docs(doc_list)
    queries,labels = generate_queries_and_labels(doc_list)
    
    positive_tokens,negative_tokens = create_pairs(queries['input_ids'],labels)
    positive_masks,negative_masks = create_pairs(queries['input_ids'],labels)
    
    data = {'positive_tokens':torch.stack(positive_tokens),
            'positive_masks':torch.stack(positive_masks),
            'negative_tokens':torch.stack(negative_tokens),
            'negative_masks':torch.stack(negative_masks)
           }

    
    return data


def evaluate(data, k=10):
    
    list_data = data

    correct_retrievals = 0
    summed_avg_precision = 0.0
    summed_reciprocal_rank = []

    for data in list_data:
        gold_ids = [pos_cntxt['id'] for pos_cntxt in data['positive_contexts']]
        number_relevant_docs = len(gold_ids)
        # check if correct doc in retrieved docs
        found_relevant_doc = False
        relevant_docs_found = 0
        current_avg_precision = 0.0

        for doc_idx, doc in enumerate(data['retrieved_contexts'][:k]):
            if doc['id'] in gold_ids:
                relevant_docs_found += 1
                if not found_relevant_doc:
                    correct_retrievals += 1
                    summed_reciprocal_rank.append(1 / (doc_idx + 1))
                current_avg_precision += relevant_docs_found / (doc_idx + 1)
                found_relevant_doc = True
                if relevant_docs_found == number_relevant_docs:
                    break
        if found_relevant_doc:
            all_relevant_docs = len(set(gold_ids))
            summed_avg_precision += current_avg_precision / all_relevant_docs

    number_of_questions = len(list_data)
    recall = correct_retrievals / number_of_questions
    mean_reciprocal_rank = sum(summed_reciprocal_rank) / number_of_questions
    mean_avg_precision = summed_avg_precision / number_of_questions
    
    
    metrics = {'k' : k,
               'recall' : recall,
               'MRR' : mean_reciprocal_rank ,
               'MAP' : mean_avg_precision ,  
               }
    
    return metrics




