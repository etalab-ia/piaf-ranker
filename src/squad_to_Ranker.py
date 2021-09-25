import json
from pathlib import Path


from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.pipeline import Pipeline
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.retriever.dense import EmbeddingRetriever
from haystack.retriever.sparse import ElasticsearchRetriever

from utils.elasticsearch_management import (delete_indices,
                                            launch_ES,
                                            prepare_mapping)


from config.dataset_creation_config import parameters,SQUAD_MAPPING
from utils.dataset_utils import extract_documents_and_labels,create_ranker_dataset,split_dataset
from utils.utils import evaluate



#launch Elasticsearch
launch_ES()


#Load parameters
evaluation_data = Path(parameters["squad_dataset"])
retriever_type = parameters["retriever_type"]
k = parameters["k"]
title_boosting_factor = parameters["boosting"]
preprocessing = parameters["preprocessing"]
split_by = parameters["split_by"]
split_length = parameters["split_length"]
split_respect_sentence_boundary = parameters["split_respect_sentence_boundary"]



# indexes for the elastic search
doc_index = "document_xp"
label_index = "label_xp"


# deleted indice for elastic search to make sure mappings are properly passed
delete_indices(index=doc_index)
delete_indices(index=label_index)


if preprocessing:
    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        clean_header_footer=False,
        split_by=split_by,
        split_length=split_length,
        split_overlap=0,  # this must be set to 0 at the data of writting this: 22 01 2021
        split_respect_sentence_boundary=False,  # the support for this will soon be removed : 29 01 2021
        )
else:
    preprocessor = None

#initialize pipeline
p = Pipeline()

#initialize pipeline elements

document_store = ElasticsearchDocumentStore(
            host="localhost",
            username="",
            password="",
            index=doc_index,
            search_fields=["name", "text"],
            create_index=False,
            embedding_field="emb",
            scheme="",
            embedding_dim=768,
            excluded_meta_data=["emb"],
            similarity="cosine",
            custom_mapping=SQUAD_MAPPING,
        )

retriever = ElasticsearchRetriever(document_store=document_store)
p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])



#load documents to document store

document_store.add_eval_data(
    evaluation_data.as_posix(),
    doc_index=doc_index,
    label_index=label_index,
    preprocessor=preprocessor,
)



#extract BM25 documents
labels,docs = extract_documents_and_labels(pipeline = p,
                                           document_store = document_store,
                                           top_k = parameters['k'],
                                           label_index = label_index,
                                           doc_index = doc_index)


dataset = create_ranker_dataset(labels,docs)
results = evaluate(dataset,k=parameters['k'])

results.update({'#_questions':len(dataset),
               })
for k,v in results.items():
    print(k,':',v)


    
#delete repeated questions
dataset = delete_repeated_questions(dataset)

#split data
train,dev,test = split_dataset(dataset)


with open('../data/raw/ranker_train.json', 'w') as fp:
       json.dump(train, fp)
        
with open('../data/raw/ranker_dev.json', 'w') as fp:
       json.dump(dev, fp)
               
with open('../data/raw/ranker_test.json', 'w') as fp:
       json.dump(test, fp)



        




