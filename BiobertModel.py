import os
import pickle
import logging
import operator
import collections
from datetime import datetime, timedelta

from GTT.Model import Model

"""
class definition of a single article input representation, to be used for Named Entity Recognition using Biobert
"""
class BiobertModel(Model):
    def __init__(self, record=None):
        Model.__init__(self)
        self.logger = logging.Logger(__name__)
        
        self.tokens_title = None
        self.tokens_abstract = None
        self.prediction_title = None
        self.prediction_abstract = None

        self.title_entities = None
        self.abstract_entities = None

        if (record):
            self.digest(record)
            self.add_to_history("biobert")
    
    """
    method to store biobert output to Document
    """
    def get_biobert_output(self, tokens_title, tokens_abstract, prediction_title, prediction_abstract):
        self.tokens_title = tokens_title
        self.tokens_abstract = tokens_abstract
        self.prediction_title = prediction_title
        self.prediction_abstract = prediction_abstract

    """
    method to assign NER results from BioBert to a single Document
    """
    def recognize(self):
        return

    """
    method to prioritize mentioned entites found in titles and abstracts with BioBert
    """
    def prioritize(self, title_weight = 5):
        return

    """
    method to prioritize mentioned entities. 
    
    each mention in abstract: 1 point
    location rank in abstract: add 1 for every point
    same in title except weighted
    """
    def prioritize_entities(self, entity_name, title_weight = 5):
        return