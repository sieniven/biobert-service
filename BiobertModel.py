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
        self.logger = logging.Logger(__name__)

        if (record):
            self.digest(record)
            self.add_to_history("biobert")

    """
    method to assign NER results from BioBert to a single Document
    """
    def recognize(self, title_entities, abstract_entities):
        self.title_entities = title_entities
        self.abstract_entities = abstract_entities

    """
    method to prioritize mentioned entites found in titles and abstracts with BioBert
    """
    def prioritize(self, title_weight = 5):
        return None

    """
    method to prioritize mentioned entities. 
    
    each mention in abstract: 1 point
    location rank in abstract: add 1 for every point
    same in title except weighted
    """
    def prioritize_entities(self, entity_name, title_weight = 5):
        return None