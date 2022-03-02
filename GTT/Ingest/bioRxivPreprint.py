#!/usr/bin/env python

# Ingests a single bioRxiv preprint


import json                                 # to convert python dictionary into desired json output format
import logging                              # to log module processes

from GTT.Model import Model
from GTT.utils import strip_tags

class bioRxivPreprint(Model):
    """A representation of a single bioRxiv article for/during ingestion.

    Inherits from Model. 

    Uses the REST API defined  at https://api.biorxiv.org/

"""
    def __init__(self, record=None):
        """ Ingests metadata of a single preprint from bioRxiv if passed
        @args
          record: a bioRxiv record returned from the REST interface.
        """
        Model.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.source = 'biorxiv'
        if (record):
            self.ingest(record)


    ##################
    # Core Functions #
    ##################

    def ingest(self,record):
        """Ingest a single bioRxiv preprint
        @args
           record: a single bioRxivPrePrint record from the REST API
        @returns
           An ingested record model
        """
        try:  
            self.id = "bioRxiv:{}".format(record['doi'])
            self.article_ids['biorxiv'] = record['doi']
        except:
            self.logger.warning("No bioRxiv ID available! {}".format(str(record)))
            self.addError()
            return None
        ###### DOI
        self.article_ids['doi'] = record['doi']
        try:
            self.article_data['title'] = record['title']
        except KeyError:
            self.logger.warning("Entry {}: No title available!".format(self.id))
            self.addError()
        ##### Abstract
        try:
            self.article_data['abstract'] = strip_tags(record['abstract'])
        except KeyError:
            self.logger.warning("Entry {}: No abstract available!".format(self.id))
            self.addError()
        try:
            journal = record['published']
            self.logger.debug("Entry {}: Journal = {}".format(self.id, journal))
            self.article_data["journal"] = journal
            if (journal != "NA"):
                self.logger.debug("Entry {}: Journal citations retrieved successfully!".format(self.id))
        except KeyError:
            self.logger.debug("Entry {}: Missing journal information!".format(self.id))
        self.mark_ingested()
            


        



### filters

def isUnPublished(article):
        return bool(article.article_data['journal'] == "NA")
    
