parameters = {
    "k": 10,
    "retriever_type": 'bm25',
    "squad_dataset": "../data/SQuAD-v1.1-dev_fr_ss999_awstart2_net.json",  
    "filter_level": None,
    "boosting": 1,
    "preprocessing": False,
    "split_by": "word",  # Can be "word", "sentence", or "passage"
    "split_length": 512,
    "split_respect_sentence_boundary": True,
}



ANALYZER_DEFAULT = {
    "analysis": {
        "filter": {
            "french_elision": {
                "type": "elision",
                "articles_case": True,
                "articles": [
                    "l",
                    "m",
                    "t",
                    "qu",
                    "n",
                    "s",
                    "j",
                    "d",
                    "c",
                    "jusqu",
                    "quoiqu",
                    "lorsqu",
                    "puisqu",
                ],
            },
            "french_stop": {"type": "stop", "stopwords": "_french_"},
            "french_stemmer": {"type": "stemmer", "language": "light_french"},
        },
        "analyzer": {
            "default": {
                "tokenizer": "standard",
                "filter": [
                    "french_elision",
                    "lowercase",
                    "french_stop",
                    "french_stemmer",
                ],
            }
        },
    }
}


SQUAD_MAPPING = {
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "text": {"type": "text"},
            "emb": {"type": "dense_vector", "dims": 512},
        },
        "dynamic_templates": [
            {
                "strings": {
                    "path_match": "*",
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"},
                }
            }
        ],
    },
    "settings": ANALYZER_DEFAULT,
}

