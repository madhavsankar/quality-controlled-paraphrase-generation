import datasets
import json
from tqdm.auto import tqdm
from torch import cuda
from math import ceil
import spacy
import pandas as pd
from functools import partial
from Bio import Align
import os
import shutil
import inspect
from urllib import request
from spacy.lang.en import English

# DocumentSemanticDiversity
import itertools
from itertools import chain
import torch
import numpy as np
from sklearn.decomposition import PCA
import faiss
import transformers
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from gensim.utils import tokenize
import gensim.downloader
from nltk.corpus import stopwords
import amrlib

def to_batches(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DocumentSemanticDiversity(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        self.config = {
            # TextDiversity configs
            'q': 1,
            'normalize': False,
            'distance_fn': faiss.METRIC_INNER_PRODUCT, # used for cosine similarity 
            'dim_reducer': PCA,
            'remove_stopwords': False, 
            'scale_dist': None, 
            'power_reg': False, 
            'mean_adj': False,
            'verbose': False,
            # DocumentSemanticDiversity configs
            'MODEL_NAME': "princeton-nlp/sup-simcse-roberta-large", # "bert-large-nli-stsb-mean-tokens",
            'use_cuda': True,
            'n_components': None
        }
        self.device = torch.device('cuda' if self.config['use_cuda'] and torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(self.config['MODEL_NAME'], device=self.device)

    def extract_features(self, corpus, return_ids=False):

        boe = np.stack(self.model.encode(corpus))
        
        # compress embedding to speed up similarity matrix computation
        if self.config['n_components'] == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.config['verbose']:
                print('Using n_components={}'.format(str(n_components)))
        else:
            n_components = -1

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = self.config['dim_reducer'](n_components=n_components).fit_transform(boe)

        if return_ids:
            ids = list(range(len(boe))) # text_ids == sentence_ids, return both
            return boe, corpus, ids, ids
        return boe, corpus

    def similarity_search(self, query_features, corpus_features, distance_fn, postprocess_fn=None):

        if postprocess_fn == "exp":
            postprocess_fn = negative_exponentiation
        elif postprocess_fn == "invert":
            postprocess_fn = complement
        else:
            postprocess_fn = None

        num_features, dims = corpus_features.shape
        dims = int(dims)
        if distance_fn == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(corpus_features) 
            if len(query_features.shape) == 1:
                query_features = np.expand_dims(query_features, 0)
            faiss.normalize_L2(query_features) 

        index = faiss.IndexFlat(dims, distance_fn) 
        index.add(corpus_features)
        D, I = index.search(query_features, num_features) 
        D = postprocess_fn(D) if postprocess_fn is not None else D
        return D[0]

    def calculate_similarity_vector(self, q_feat, c_feat):

        z = self.similarity_search(query_features=q_feat, 
                              corpus_features=c_feat,
                              distance_fn=self.config['distance_fn'], 
                              postprocess_fn=self.config['scale_dist'])

        # remove some noise from the z similarities
        if self.config['power_reg']:
            z **= 2

        if self.config['mean_adj']:
            z -= z.mean()
            z = np.where(z < 0, 0 , z)

        return z

    def find_similarity(self, pair):
        query = pair[0]
        corpus = pair[1]

        # extract features + species
        feats, corpus = self.extract_features([query, corpus])
        q_feats, q_corpus = feats[0], corpus[0]
        c_feats, c_corpus = feats[1:], corpus[1:]

        # if there are no features, we cannot rank
        if len(q_feats) == 0 or len(c_feats) == 0:
            return 0

        # get similarity vector z
        z = self.calculate_similarity_vector(q_feats, c_feats)

        return z[0]

    def _compute(self, predictions, references, batch_size=64, device=None):
        """Returns the scores"""

        # preds_embeds = []
        # refs_embeds = []
        # for preds, refs in tqdm(zip(to_batches(predictions, batch_size), to_batches(references, batch_size)), total=int(ceil(min(len(predictions),len(references)) / batch_size)), desc="phon_metric"):
            
        #     preds_embeds.append(self.embed(preds))
        #     refs_embeds.append(self.embed(refs)) 
        
        # preds = torch.cat(preds_embeds)
        # refs = torch.cat(refs_embeds)

        scores = list(tqdm(map(self.find_similarity, zip(predictions, references)), total=len(predictions), desc="semantic:find_sim"))

        return {
            "scores": scores,
        }