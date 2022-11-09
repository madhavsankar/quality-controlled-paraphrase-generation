import datasets
from transformers import AutoTokenizer, AutoModel
import json
from tqdm.auto import tqdm
from torch import cuda
import torch
from math import ceil
import spacy
import numpy as np
from sklearn.decomposition import PCA
from Bio import Align
import os
import shutil
import inspect
from urllib import request
import itertools
from spacy.lang.en import English

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

def find_max_list(lists):
    list_len = [len(l) for l in lists]
    return max(list_len)

def is_list_of_lists(input_list):
    return any(isinstance(el, list) for el in input_list)

def tag2alpha(tags):
    # build dict of unique tags
    tag_map = set(itertools.chain(*tags))
    tag_map = {tag: chr(i+65) for i, tag in enumerate(tag_map)}
    # apply to tags
    if isinstance(tags, np.ndarray):
        tags_to_alpha_fn = np.vectorize(tag_map.get)
        tags = tags_to_alpha_fn(tags)
    else:
        tags = [list(map(tag_map.get, tag)) for tag in tags]
    return tags

def align_and_score(seq1, seq2):
    return Align.PairwiseAligner().align(seq1, seq2).score


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
class MorphMetric(datasets.Metric):
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
            # POSSequenceDiversity configs
            'pad_to_max_len': False, 
            'use_gpu': False,
            'n_components': None,
            'split_sentences': False,
        }

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

        # extracts parts-of-speech (poses)
        poses = []
        for s in corpus:
            pos = [token.pos_ for token in self.model(s)] #if token.text not in stopwords]
            poses.append(pos)

        # compute max seq length for padding / normalization later
        self.max_len = find_max_list(poses)

        # pad to max sentence doc length
        if self.config['pad_to_max_len']:
            poses = np.array([s + ['NULL'] * (self.max_len - len(s)) for s in poses])

        if return_ids:
            return poses, corpus, text_ids, sentence_ids 
        return poses, corpus

    def calculate_similarity_vector(self, q_feat, c_feat):

        features = [q_feat] + c_feat

        if is_list_of_lists(features):
            # convert pos tags to alphas
            features = tag2alpha(features)
            features = ["".join(pos) for pos in features]

        q_feat = features[0]
        c_feat = features[1:]

        q_len = len(q_feat)
        scores = []
        for f in c_feat:
            score = align_and_score(q_feat, f)
            score /= max(len(f), q_len)
            scores.append(score)

        z = np.array(scores)

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
        # for preds, refs in tqdm(zip(to_batches(predictions, batch_size), to_batches(references, batch_size)), total=int(ceil(min(len(predictions),len(references)) / batch_size)), desc="morph_metric"):
            
        #     preds_embeds.append(self.embed(preds))
        #     refs_embeds.append(self.embed(refs)) 
        
        # preds = torch.cat(preds_embeds)
        # refs = torch.cat(refs_embeds)

        scores = list(tqdm(map(self.find_similarity, zip(predictions, references)), total=len(references), desc="morphmetric:find_sim"))

        return {
            "scores": scores,
        }