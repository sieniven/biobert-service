#!/usr/bin/env python

__author__ = 'Greg Tucker-Kellogg'

"""
@package PubMedArticle
@author  Greg Tucker-Kellogg
@brief Ingest a PubMedArticle or PubMedBookArticle
"""


import json                                 # to convert python dictionary into desired json output format
from datetime import datetime          
import logging                              # to log module processes

from GTT.Model import Model
from GTT.utils import strip_tags

class PubMedArticle(Model):
    """A representation of a single Pubmed article for/during GenesThatTweet ingestion.

    Handles both PubmedArticle and PubmedBookArticle

    Inherits from Model. 
    
"""
    def __init__(self, record):
        """ Ingests metadata of a single paper from PubMed, passed in as a record from efetch
        @args
          record: a PubMed record returned from efetch. Should be a PubmedArticle or PubmedBookArticle
        """
        Model.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.source = 'pubmed'
        if (record):
            self.ingest(record)


    def ingest(self,record):
        """The main ingest function. 

        Handles data differently for PubMedArticle and PubMedBookArticle objects

        @args
        record: a single record from efectch or equivalent in Bio.Entrez
        """
        if (record.get('MedlineCitation')):
            self.ingestPubMedArticle(record)
        elif (record.get('BookDocument')):
            self.ingestPubMedBookArticle(record)
        else:
            self.logger.debug("Unknown record: {}".format(json.dumps(record, indent=4)))
            self.record = None


    def ingestPubMedBookArticle(self,record):
        """Ingest a single PubMed book chapter
        @args
           record: a single PubMedBookArticle record from Entrez.efetch()
        @returns
           An ingested record model
        """
        try:  
            self.pmid = str(record['BookDocument']['PMID'])
            self.id = f"PMID:{self.pmid}"
            self.article_ids['pubmed'] = self.pmid
        except:
            self.logger.warning(f"No PubMed ID available! {record['BookDocument']}")
            self.addError()
            return None
        ###### DOI
        for e in record['BookDocument']['Book']['ELocationID']:
            if (e.attributes['EIdType'] == "doi"):
                self.article_ids['doi'] = str(e)
        if (self.article_ids['doi'] == None):
            self.logger.debug(f"Entry {self.pmid}: No ELocationID available!")
        ##### Title 
        try:
            self.article_data['title'] = strip_tags(str(record['BookDocument']['ArticleTitle']))
        except KeyError:
            self.logger.warning(f"Entry {self.pmid}: No title available!")
            self.addError()
        ##### Abstract
        try:
            self.article_data['abstract'] = strip_tags(" ".join(record['BookDocument']['Abstract']['AbstractText']))
        except KeyError:
            self.logger.debug(f"Entry {self.pmid}: No abstract available!")
            self.addError()
        try:
            BookTitle = str(record['BookDocument']['Book']['BookTitle'])
            Publisher = str(record['BookDocument']['Book']['Publisher']['PublisherName'])
            PubDate = str(record['BookDocument']['Book']['PubDate']['Year'])
            Book = "{} ({}, {})".format(BookTitle,Publisher,PubDate)
            self.logger.debug("Entry {}: Book = {}".format(self.pmid, Book))
            self.article_data["journal"] = Book
            self.logger.debug("Entry {}: Book citation retrieved successfully!".format(self.pmid))                             
        except KeyError:
            self.logger.debug("Entry {}: Missing Book information!".format(self.pmid))
        self.mark_ingested()


        
            
    def ingestPubMedArticle(self,record):
        """Ingest a single PubMedArticle
        @args
           record: a single PubMedArticle record from Entrez.efetch()
        @returns
           An ingested record model
        """
        try:  
            self.pmid = str(record['MedlineCitation']['PMID'])
            self.id = f"PMID:{self.pmid}"
            self.article_ids['pubmed'] = self.pmid
            article = record['MedlineCitation']['Article']
        except:
            self.logger.warning("No PubMed ID available! {}".format(record['MedlineCitation']))
            self.addError()
            return None
        ###### DOI
        for e in article['ELocationID']:
            if (e.attributes['EIdType'] == "doi"):
                self.article_ids['doi'] = str(e)
        if (self.article_ids['doi'] == None):
            self.logger.debug(f"Entry {self.pmid}: No ELocationID available!")
        ##### Title 
        try:
            self.article_data['title'] = strip_tags(str(article['ArticleTitle']))
        except KeyError:
            self.logger.warning(f"Entry {self.pmid}: No title available!")
            self.addError()
        ##### Abstract
        try:
            self.article_data['abstract'] = strip_tags(" ".join(article['Abstract']['AbstractText']))
        except KeyError:
            self.logger.debug(f"Entry {self.pmid}: No abstract available!")
            self.addError()
        try:
            j_title = article["Journal"]["Title"]
            j_volume = article["Journal"]["JournalIssue"]["Volume"]
            j_issue = article["Journal"]["JournalIssue"]["Issue"]
            j_pages = article["Pagination"]["MedlinePgn"]
            journal = j_title + ", " + j_volume + "(" + j_issue + "), " + j_pages
            self.logger.debug("Entry {}: Journal = {}".format(self.pmid, journal))
            self.article_data["journal"] = journal
            self.logger.debug("Entry {}: Journal citations retrieved successfully!".format(self.pmid))                             
        except KeyError:
            self.logger.debug("Entry {}: Missing journal information!".format(self.pmid))
        self.mark_ingested()
            




