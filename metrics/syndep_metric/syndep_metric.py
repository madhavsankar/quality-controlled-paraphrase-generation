import datasets
from transformers import AutoTokenizer, AutoModel
import json
from tqdm.auto import tqdm
from torch import cuda
import torch
from math import ceil
import spacy
import numpy as np
import pandas as pd
from functools import partial
from Bio import Align
import os
import shutil
import inspect
from urllib import request
import itertools
from spacy.lang.en import English

# tree libs
from sklearn.decomposition import PCA
import networkx as nx
from karateclub import FeatherGraph, GL2Vec, Graph2Vec, LDP
import zss

def to_batches(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def clean_text(texts):
    texts = [text.replace("<br />", "") for text in texts]
    texts = [text.replace("...", ". ") for text in texts]
    texts = [text.replace("..", ". ") for text in texts]
    texts = [text.replace(".", ". ") for text in texts]
    texts = [text.replace("!", "! ") for text in texts]
    texts = [text.replace("?", "? ") for text in texts]
    texts = [text.strip() for text in texts]
    return texts

def split_sentences(texts, return_ids=False):
    sentencizer = English()
    sentencizer.add_pipe("sentencizer")

    if isinstance(texts, str):
        texts = [texts]

    sentences, text_ids, sentence_ids = [], [], []
    for text_id, text in enumerate(texts):
        sents = list(sentencizer(text).sents)
        sents = [s.text.strip() for s in sents if s.text.strip()]
        sentences.extend(sents)
        text_ids.extend([text_id] * len(sents))
        sentence_ids.extend(list(range(len(sents))))
    
    if return_ids:
        return sentences, text_ids, sentence_ids
    return sentences

def node_match_on_pos(G1_node, G2_node):
    return G1_node['pos'] == G2_node['pos']

def node_match_on_labels(G1_node, G2_node):
    return G1_node['labels'].sort() == G2_node['labels'].sort()

def edge_match_on_dep(G1_edge, G2_edge):
    return G1_edge['dep'] == G2_edge['dep']

# used with zss ====================================================================
def get_nodes_dict(T):
    nodes_dict = {}
    for edge in T.edges():
        if edge[0] not in nodes_dict:
            nodes_dict[edge[0]] = zss.Node(edge[0])
        if edge[1] not in nodes_dict:
            nodes_dict[edge[1]] = zss.Node(edge[1])
        nodes_dict[edge[0]].addkid(nodes_dict[edge[1]])
    return nodes_dict

def zss_tree_edit_distance(G1, G2):
    source1 = [n for (n, d) in G1.in_degree() if d == 0][0]
    T1 = nx.dfs_tree(G1, source=source1)
    T1_nodes_dict = get_nodes_dict(T1)
    
    source2 = [n for (n, d) in G2.in_degree() if d == 0][0]
    T2 = nx.dfs_tree(G2, source=source2)
    T2_nodes_dict = get_nodes_dict(T2)

    return zss.simple_distance(T1_nodes_dict[source1], T2_nodes_dict[source2])

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

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
class DependencyDiversity(datasets.Metric):
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
        self.model = spacy.load("en_core_web_sm")
        self.config = {
            # TextDiversity configs
            'q': 1,
            'normalize': False,
            'dim_reducer': PCA,
            'remove_stopwords': False, 
            'verbose': False,
            # DependencyDiversity configs
            'similarity_type':"ldp",
            'use_gpu': False,
            'n_components': None,
            'split_sentences': False,
        }
    
    def generate_dependency_tree(self, in_string):
        ''' 
        NOTES: 
          - Use the index instead of the token to avoid loops
          - Ensure the node is a string since zss tree edit distance requires it
        '''
        doc = self.model(in_string)

        G = nx.DiGraph()

        nodes = [(str(token.i), {'text': token.text, 'pos' : token.pos_}) 
                 for token in doc 
                 # if not token.is_punct
        ]
        G.add_nodes_from(nodes)

        edges = [(str(token.head.i), str(token.i), {'dep' : token.dep_}) 
                 for token in doc 
                 if token.head.i != token.i 
                 # and not token.is_punct
                 # and not token.head.is_punct
        ]
        G.add_edges_from(edges)

        return G

    def extract_features(self, corpus, return_ids=False):

        # clean corpus
        corpus = clean_text(corpus)

        # split sentences
        if self.config['split_sentences']:
            corpus, text_ids, sentence_ids = split_sentences(corpus, return_ids=True)
        else:
            ids = list(range(len(corpus)))
            text_ids, sentence_ids = ids, ids

        # # remove any blanks...
        # corpus = [d for d in corpus if len(d.strip()) > 0]

        # generate dependency tree graphs
        features = [self.generate_dependency_tree(d) for d in corpus]

        # optionally embed graphs
        if 'distance' not in self.config['similarity_type']:
        
            # the embedding approaches require integer node labels
            features = [nx.convert_node_labels_to_integers(g, first_label=0, ordering='default') for g in features]

            if self.config['similarity_type'] == "ldp":
                model = LDP(bins=64) # more bins, less similarity
                model.fit(features)
                emb = model.get_embedding().astype(np.float32)
            elif self.config['similarity_type'] == "feather":
                model = FeatherGraph(theta_max=100) # higher theta, less similarity
                model.fit(features)
                emb = model.get_embedding().astype(np.float32)

            # compress embedding to speed up similarity matrix computation
            if self.config['n_components'] == "auto":
                n_components = min(max(2, len(emb) // 10), emb.shape[-1])
                if self.config['verbose']:
                    print('Using n_components={}'.format(str(n_components)))
            else:
                n_components = -1

            if type(n_components) == int and n_components > 0 and len(emb) > 1:
                emb = self.config['dim_reducer'](n_components=n_components).fit_transform(emb)

            features = emb

        if return_ids:
            return features, corpus, text_ids, sentence_ids
        return features, corpus

    def calculate_similarity_vector(self, q_feat, c_feat):

        if 'distance' in self.config['similarity_type']:

            if self.config['similarity_type'] == "graph_edit_distance":
                dist_fn = partial(nx.graph_edit_distance, 
                             node_match=node_match_on_pos, 
                             edge_match=edge_match_on_dep)
            elif self.config['similarity_type'] == "tree_edit_distance":
                dist_fn = zss_tree_edit_distance
                
            z = np.array([dist_fn(q_feat, f) for f in c_feat])

            # convert distance to similarity
            z = z - z.mean()
            z = 1 / (1+np.e**z)

        else:
            z = np.array([cos_sim(q_feat, f).item() for f in c_feat])

            # strongly penalize for any differences to make Z more intuitive
            z **= 200

        return z

    def find_diversity(self, pair):
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

        return (1 - z[0])

    def _compute(self, predictions, references, batch_size=64, device=None):
        """Returns the scores"""

        # preds_embeds = []
        # refs_embeds = []
        # for preds, refs in tqdm(zip(to_batches(predictions, batch_size), to_batches(references, batch_size)), total=int(ceil(min(len(predictions),len(references)) / batch_size)), desc="phon_metric"):
            
        #     preds_embeds.append(self.embed(preds))
        #     refs_embeds.append(self.embed(refs)) 
        
        # preds = torch.cat(preds_embeds)
        # refs = torch.cat(refs_embeds)

        scores = list(tqdm(map(self.find_diversity, zip(predictions, references)), total=len(predictions), desc="syndep:find_sim"))

        return {
            "scores": scores,
        }