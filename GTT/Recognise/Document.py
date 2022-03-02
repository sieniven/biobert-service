#!/usr/bin/env python

__author__ = 'Greg Tucker-Kellogg'

"""
@package Document
@author  Greg Tucker-Kellogg
@brief A document during named entity recognition
"""


import logging                              # to log module processes
import operator
from GTT.Model import Model
from GTT.Recognise.Reach import Reach


class Document(Model):
    """A representation of a single Document for Named Entity Recognition (Identification)"""

    def __init__(self, record=None):
        """ Ingests data of a single document
        @args
          record: a Document record for the Identification process
        """
        Model.__init__(self)
        self.logger = logging.getLogger(__name__)
        if (record):
            self.digest(record)
            self.recognise()
            self.prioritise()
            self.add_to_history('normalised')

    def recognise(self):
        """Assign NER results from Reach"""
        self.titleEntities = Reach(self.article_data['title'])
        self.abstractEntities = Reach(self.article_data['abstract'])

    def prioritise(self,titleWeight=5,keepUAZ=False):
        self.scores = {}
        scored_proteins = self.prioritise_entities('proteins',titleWeight)
        scored_families = self.prioritise_entities('families',titleWeight)
        scores = {**scored_proteins, **scored_families}
        if not(keepUAZ):
            scores = {k:v for k,v in scores.items() if (v['namespace'] != 'uaz')}
        self.entities = scores
        self.tags['normalised'] = True

    def prioritise_entities(self,entity_name,titleWeight=5):
        """Assign priority to mentioned entities.

        - each mention in abstract: 1 point
        - location rank in abstract: add 1 for every point
        - same in title except weighted
        """

        if (entity_name == 'proteins'):
            scored_entities = self.abstractEntities.proteins
            SCORED_ENTITIES = self.titleEntities.proteins
        elif (entity_name == 'families'):
            scored_entities = self.abstractEntities.families
            SCORED_ENTITIES = self.titleEntities.families
        else:
            return None

        # THE ABSTRACTS
        # the counts
        for k, _ in scored_entities.items():
            scored_entities[k]['score'] = scored_entities[k]['count']
            scored_entities[k]['abstract_count'] = scored_entities[k]['count']

        # the locations
        positions = {k: v['start_pos'] for k, v in scored_entities.items()}
        positions = sorted(positions.items(), key=operator.itemgetter(1),reverse=True)
        for i in range(len(positions)):
            k = positions[i][0]
            scored_entities[k]['score'] = scored_entities[k]['score'] + i
            scored_entities[k]['abstract_position_score'] = i

        # THE TITLES
        # the counts
        for k, v in SCORED_ENTITIES.items():
            if (scored_entities.get(k)):
                if (scored_entities[k].get('score')):
                    scored_entities[k]['score'] = (scored_entities[k]['score'] + SCORED_ENTITIES[k]['count'] * titleWeight)
                    scored_entities[k]['count'] = (scored_entities[k]['count'] + SCORED_ENTITIES[k]['count'] * titleWeight)
                else:
                    scored_entities[k]['score'] = SCORED_ENTITIES[k]['count'] * titleWeight
            else:
                scored_entities[k] = SCORED_ENTITIES[k]
                scored_entities[k]['score'] = SCORED_ENTITIES[k]['count'] * titleWeight
            scored_entities[k]['title_count'] = SCORED_ENTITIES[k]['count']

        # the locations
        positions = {k: v['start_pos'] for k, v in SCORED_ENTITIES.items()}
        positions = sorted(positions.items(), key=operator.itemgetter(1),reverse=True)
        for i in range(len(positions)):
            k = positions[i][0]
            scored_entities[k]['score'] = scored_entities[k]['score'] + i * titleWeight
            scored_entities[k]['title_position_score'] = i

        # the results
        return scored_entities

    def prioritise_proteins(self):
        """Assign priority to mentioned entities.

        Start with proteins
        - mention in title: 2 points
        - each mention in abstract: 1 point
        - location rank in abstract: add 1 for every point
        """
        # THE ABSTRACTS
        proteins = self.abstractEntities.proteins
        # the counts
        for k, _ in proteins.items():
            proteins[k]['score'] = proteins[k]['count']

        # the locations
        positions = {k: v['start_pos'] for k, v in proteins.items()}
        positions = sorted(positions.items(), key=operator.itemgetter(1))
        for i in range(len(positions)):
            k = positions[i][0]
            proteins[k]['score'] = proteins[k]['score'] + i

        # THE TITLES
        PROTEINS = self.titleEntities.proteins
        titleWeight = 3
        # the counts
        for k, v in PROTEINS.items():
            if (proteins.get(k)):
                if (proteins[k].get('score')):
                    proteins[k]['score'] = (proteins[k]['score'] + PROTEINS[k]['count'] * titleWeight)
                else:
                    proteins[k]['score'] = PROTEINS[k]['count'] * titleWeight
            else:
                proteins[k] = PROTEINS[k]
                proteins[k]['score'] = PROTEINS[k]['count'] * titleWeight
        self.protein_scores = proteins

        # the locations
        positions = {k: v['start_pos'] for k, v in PROTEINS.items()}
        positions = sorted(positions.items(), key=operator.itemgetter(1))
        for i in range(len(positions)):
            k = positions[i][0]
            proteins[k]['score'] = proteins[k]['score'] + i * titleWeight
        self.proteinscores = proteins

    def prioritise_families(self):
        """Assign priority to mentioned entities.

        - mention in title: 2 points
        - each mention in abstract: 1 point
        - location rank in abstract: add 1 for every point
        """
        # THE ABSTRACTS
        families = self.abstractEntities.families
        # the counts
        for k, v in families.items():
            v['score'] = v['count']

        # the locations
        positions = {k: v['start_pos'] for k, v in families.items()}
        positions = sorted(positions.items(), key=operator.itemgetter(1))
        for i in range(len(positions)):
            k = positions[i][0]
            families[k]['score'] = families[k]['score'] + i

        # THE TITLES
        FAMILIES = self.titleEntities.families
        titleWeight = 3
        for k, v in FAMILIES.items():
            if (families.get(k)):
                if (families[k].get('score')):
                    families[k]['score'] = (families[k]['score'] + FAMILIES[k]['count'] * titleWeight)
                else:
                    families[k]['score'] = FAMILIES[k]['count'] * titleWeight
            else:
                families[k] = FAMILIES[k]
                families[k]['score'] = FAMILIES[k]['count'] * titleWeight
        self.family_scores = families

    def update(self):
        self.logger.warning("About to update")
        new_values = {"$set": {"entities": self.entities,
                               "history": self.history,
                               "tags": self.tags}}
        Model.update(self,new_values)
