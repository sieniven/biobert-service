import os
import sys
import pickle
import logging
import operator
import collections
from datetime import datetime, timedelta
import gilda
from indra.databases import uniprot_client

sys.path.append("../")
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
                entity = ""
                for tok in self.tokens_title[1][start_index:end_index]:
                    if tok.startswith("##"):
                        entity += tok[2:]
                    else:
                        entity += " " + tok
                entity = entity[1:]
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
                entity = ""
                for tok in self.tokens_abstract[1][start_index:end_index]:
                    if tok.startswith("##"):
                        entity += tok[2:]
                    else:
                        entity += " " + tok
                entity = entity[1:]
                self.abstract_entities.append(entity)
                named_entity_found = False
            elif named_entity_found:
                continue

    def get_proteins(self):
        uids = {}
        for entry in self.abstract_entities:
            scored_matches = gilda.ground(entry)
            if scored_matches:
                _data = scored_matches[0].to_json()
                key = _data['term']['id']
                ## see if uniprot db exists
                if (scored_matches[0].term.db == 'UP') and (scored_matches[0].term.organism == '9606'): #human
                    hgnc_id = uniprot_client.get_hgnc_id(key)
                    entrez_id = uniprot_client.get_entrez_id(key)
                    symbol = uniprot_client.get_gene_name(key)
                    namespace = "uniprot"
                elif (scored_matches[0].term.db == 'HGNC'):
                    hgnc_id = _data['term']['id']
                    symbol = _data['term']['entry_name']
                    namespace = 'HGNC'
                    entrez_id = ""
                else:
                    hgnc_id = ""
                    symbol = ""
                    entrez_id = ""
                    namespace = ""

                if key in uids:
                    uids[key]['count'] = uids[key]['count'] + 1
                else:
                    uids[key] = {"count": 1,
                                #"text": entry["text"],
                                #"type": entry["type"],
                                "namespace": namespace,
                                "hgnc_id": hgnc_id,
                                "entrez_id": entrez_id,
                                "symbol": symbol
                                }
        self.abstract_proteins = uids

        uids = {}
        for entry in self.title_entities:
            scored_matches = gilda.ground(entry)
            if scored_matches:
                _data = scored_matches[0].to_json()
                key = _data['term']['id']
                ## see if uniprot db exists
                if (scored_matches[0].term.db == 'UP') and (scored_matches[0].term.organism == '9606'): #human
                    hgnc_id = uniprot_client.get_hgnc_id(key)
                    entrez_id = uniprot_client.get_entrez_id(key)
                    symbol = uniprot_client.get_gene_name(key)
                    namespace = "uniprot"
                elif (scored_matches[0].term.db == 'HGNC'):
                    hgnc_id = _data['term']['id']
                    symbol = _data['term']['entry_name']
                    namespace = 'HGNC'
                    entrez_id = ""
                else:
                    hgnc_id = ""
                    symbol = ""
                    entrez_id = ""
                    namespace = ""

                if key in uids:
                    uids[key]['count'] = uids[key]['count'] + 1
                else:
                    uids[key] = {"count": 1,
                                #"text": entry["text"],
                                #"type": entry["type"],
                                "namespace": namespace,
                                "hgnc_id": hgnc_id,
                                "entrez_id": entrez_id,
                                "symbol": symbol
                                }
        self.title_proteins = uids
        return 


    def get_families(self):
        uids = {}
        for entry in self.abstract_entities:
            scored_matches = gilda.ground(entry)
            if scored_matches and (scored_matches[0].term.db == 'FPLX'): #FAMPLEX
                _data = scored_matches[0].to_json()
                key = _data['term']['id']
                namespace = "FamPlex"

                if key in uids:
                    uids[key]['count'] = uids[key]['count'] + 1
                else:
                    uids[key] = {"count": 1,
                                #"start_pos": entry['start-pos']['offset'],
                                "namespace": namespace,
                                "symbol" : "",
                                #"text": entry["text"],
                                }
        self.abstract_families = uids

        uids = {}
        for entry in self.title_entities:
            scored_matches = gilda.ground(entry)
            if scored_matches and (scored_matches[0].term.db == 'FPLX'): #FAMPLEX
                _data = scored_matches[0].to_json()
                key = _data['term']['id']
                namespace = "FamPlex"

                if key in uids:
                    uids[key]['count'] = uids[key]['count'] + 1
                else:
                    uids[key] = {"count": 1,
                                #"start_pos": entry['start-pos']['offset'],
                                "namespace": namespace,
                                "symbol" : "",
                                #"text": entry["text"],
                                }
        self.title_families = uids
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