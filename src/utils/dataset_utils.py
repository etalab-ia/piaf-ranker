from tqdm import tqdm
import random



def extract_documents_and_labels(pipeline,
                                 document_store,
                                 top_k,
                                 label_index = "label_xp",
                                 doc_index = "document_xp",
                                 label_origin = "gold_label",
                                 ):

    """
    the function is used to extract the labels and retreived document from haystack
    input :
        - pipeline 
        - document_store
        - top_k : top k docs returned 
        
    output :
        - question_label_dict_list : list of dictionnaries with query,label docs and their ids
        - retrieved_docs_list : query list of retreived docs list with their text and ids 
    
    """
    filters = {"origin": [label_origin]}
    #extract labels
    
    labels = document_store.get_all_labels_aggregated(
        index=label_index, filters=filters
    )


    question_label_dict_list = []
    for label in labels:
        if not label.question:
            continue
        deduplicated_doc_ids = list(set([str(x) for x in label.multiple_document_ids]))
        question_label_dict = {
            "query": label.question,
            "gold_ids": deduplicated_doc_ids,
        }
        question_label_dict_list.append(question_label_dict)

    retrieved_docs_list = [
        pipeline.run(query=question["query"], top_k_retriever=top_k, index=doc_index)
        for question in question_label_dict_list
    ]


    for label in question_label_dict_list:
        docs = []
        for id in label['gold_ids']:
            doc = document_store.get_document_by_id(id,index=doc_index)
            docs.append(doc)
        label['docs']=docs
        
    
    return question_label_dict_list,retrieved_docs_list




def create_ranker_dataset(question_label_dict_list,
                          retrieved_docs_list,
                          keep_positives=True):
    
     """
    turn label_list and doc_list into a one list to save as json
    
    input :
        - question_label_dict_list : list of dictionnaries with query,label docs and their ids
        - retrieved_docs_list : query list of retreived docs list with their text and ids 
        
    output :
        - list_data : list of dictionnaries {'question','positive_contexts'=[],'retrieved_contexts'=[]}
    
    """
    list_data=[]
    
    for labels, retrieved_docs in tqdm(
            zip(question_label_dict_list, retrieved_docs_list)
        ):

        questions = ''
        negative_docs = []
        positive_docs = []

        if labels['query'].lower() == retrieved_docs['query'].lower():
            question = labels['query']
        else:
            print('label and document questions does not match' )

        gold_ids = labels['gold_ids']

        for label in labels['docs']:
            positive_docs.append({'id':label.id,
                                 'context':label.text,
                                 })

        for retrieved_doc in retrieved_docs['documents']:
            if keep_positives or retrieved_doc.id not in gold_ids:
                negative_docs.append({'id':retrieved_doc.id,
                                     'title':retrieved_doc.meta['name'],
                                     'context':retrieved_doc.text,
                                     'score':retrieved_doc.score,
                                     'probability':retrieved_doc.probability}) 



        data_dict = {'question':question,
                     'positive_contexts':positive_docs,
                     'retrieved_contexts':negative_docs,
                    }


        list_data.append(data_dict)
        
    return list_data


def delete_repeated_questions(dataset):
    
    '''
    Delete repeated queries
    
    '''

    list_questions=[]
    for index,query in enumerate(dataset):
        question = query['question'] 
        if question in list_questions:
            dataset.pop(index)
        list_questions.append(question) 
    
    return dataset



def split_dataset(list_,train_ratio = 0.8,dev_ratio = 0.1):
    '''
    Split dataset to train,dev and test sets
    
    '''
    
    random.shuffle(list_)
    trainning_index = int(len(list_) * train_ratio)
    dev_index = int(len(list_) * (train_ratio+dev_ratio))
    train = list_[:trainning_index]
    dev = list_[trainning_index:dev_index]
    test = list_[dev_index:]
    
    return train,dev,test
    
    