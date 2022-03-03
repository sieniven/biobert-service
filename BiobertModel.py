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

        self.title_entities = []
        self.abstract_entities = []

        if (record):
            self.digest(record)
            self.add_to_history("biobert")
    
    """
    method to store biobert output to Document
    """
    def get_biobert_output(self, index, tokens_title, tokens_abstract, prediction_title, prediction_abstract):
        if (index != tokens_title[0] or index != tokens_abstract[0] or 
            index != prediction_title[0] or index != prediction_abstract[0]):
            return
        else:
            self.tokens_title = tokens_title
            self.tokens_abstract = tokens_abstract
            self.prediction_title = prediction_title
            self.prediction_abstract = prediction_abstract

    """
    method to assign NER results from BioBert to a single Document
    """
    def recognize(self):
        # process predicted title
        start_index = 0
        end_index = 0
        named_entity_found = False
        for idx, predicted_label in enumerate(self.prediction_title[1]):
            if predicted_label == "B":
                start_index = idx
                named_entity_found = True
            elif predicted_label == "O" and named_entity_found:
                end_index = idx
                text = ''.join(x for x in self.tokens_title[1][start_index:end_index])
                entity = text.replace('#', '')
                self.title_entities.append(entity)
                named_entity_found = False

        # process predicted abstract
        start_index = 0
        end_index = 0
        named_entity_found = False
        for idx, predicted_label in enumerate(self.prediction_abstract[1]):
            if predicted_label == "B":
                start_index = idx
                named_entity_found = True
            elif predicted_label == "O" and named_entity_found:
                end_index = idx
                text = ''.join(x for x in self.tokens_abstract[1][start_index:end_index])
                entity = text.replace('#', '')
                self.abstract_entities.append(entity)
                named_entity_found = False
            elif named_entity_found:
                continue

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