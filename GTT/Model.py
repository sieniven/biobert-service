#!/usr/bin/env python

# The Ingestion model class

import json
import logging
from datetime import datetime

import jsonschema
from jsonschema.exceptions import ValidationError

# import GTT
# from GTT.Persist.Articles import coll as collection


class Model:
    """A mostly abstract model for GenesThatTweet data passed between services

    Each instance of this class (or any derived class) is meant to
    represent a single publication/preprint as a source for GTT.
    """

    # This is the json schema for the interface
    # schema = json.loads(GTT.schema)
    # This class variable is used to decide whether validation is meaningful
    # validation = False

    def __init__(self):
        """Initialise a Model instance.

        The initial model has enough to satisfy the data structure
        requirements for passing through the services, but no actual
        data in the slots.
        """
        self.errorCount = 0
        self.logger = logging.getLogger(__name__)
        self.source = ""
        self.id = ""
        self.article_ids = {"pubmed": "",
                            "arxiv": "",
                            "doi": "",
                            "twitter": {"tweet_id": 0,
                                        "user_id": 0,
                                        "username": ""}}
        self.article_data = {"title": "",
                             "abstract": "",
                             "body": "",
                             "journal": ""}
        self.tags = {"ingested": False,
                     "normalised": False}
        self.history = []
        self.entities = {}
        self.errors = []
        self.posthandlers = []

    def mark_ingested(self):
        '''Note that (and when) the ingestion is completed for the article/preprint.'''
        self.tags['date_ingested']: datetime.utcnow().strftime("%Y/%m/%d %H:%M")
        self.tags['ingested'] = True
        self.add_to_history("ingested")

    def ingested(self):
        '''Return the ingestion status'''
        return self.tags['ingested']

    def add_to_history(self, event):
        """Add an event to this document's history"""
        self.history.append(str(event))

    def postData(self):
        """Return the hash for posting to another service.

        The data that is transferred from service to service follows a
        defined schema. postData() provides that data for service consuption.

        """
        return {"_id": self.id,
                "article_ids": self.article_ids,
                "source": self.source,
                "article_data": self.article_data,
                "tags": self.tags,
                "entities": self.entities,
                "history": self.history
                }

    def post_one(self, handlers=[]):
        postData = self.postData()
        # if (Model.validation):
        #     try:
        #         jsonschema.validate(postData, self.__class__.schema)
        #         self.logger.debug("Validated data for posting")
        #     except ValidationError as VE:
        #         self.logger.warning(
        #             f"Failed to validate data {postData}: {VE}")
        for posthandler in filter(callable, (self.posthandlers + handlers)):
            try:
                posthandler(postData)
                self.logger.info(
                    f"posted {self.id} {type(self).__name__} record using {str(posthandler)}")
            except Exception as E:
                self.logger.error(
                    f"Failed to post {len(postData)} records using {str(posthandler)}: {E}")

    def digest(self, data):
        """digest data provided by a service,
        typically via a request handler

        Like ingest, except the data is already in
        postData form :-) """

        try:
            # if (self.__class__.validation is True):
            #     jsonschema.validate(data, self.__class__.schema)
            self.id = data["_id"]
            self.article_ids = data["article_ids"]
            self.source = data['source']
            self.article_data = data['article_data']
            self.tags = data["tags"]
            self.history = data['history']
            if "entities" in data:
                self.entities = data["entities"]
            else:
                self.entities = {}
            # if (Model.validation is True):
            #     self.validate()
        except KeyError as e:
            self.logger.error(f"failed to digest {data}: {e}")
            self.addError()
        except ValidationError as e:
            self.logger.error(f"failed to validate {data}: {e}")

    def addError(self):
        '''Increment the count of ingestion errors'''
        self.errorCount = self.errorCount + 1

    def validate(self):
        """validate the object against the main GTT model schema
        """
        # if Model.validation is True:
        #     return jsonschema.validate(self.postData(), self.__class__.schema)

    # def insert(self,overwrite_ok=False):
    #     """Insert into collection """
    #     if not collection:
    #         self.logger.info("persistence not available")
    #         return
    #     else:
    #         data = self.postData()
    #         item = {"_id": data["_id"]}
    #         if collection.find_one(item):
    #             if overwrite_ok:
    #                 collection.replace_one(data)
    #         else:
    #             collection.insert_one(data)

    # def update(self,new_values):
    #     """update in collection"""
    #     if not collection:
    #         self.logger.info("persistence not available")
    #         return
    #     else:
    #         item = {"_id": self.id}
    #         if collection.find_one(item):
    #             self.logger.info("attempting to modify one entry")
    #             UpdateResult = collection.update_one(item,new_values)
    #             self.logger.info(f"modified {UpdateResult.modified_count} entries")
    #         else:
    #             self.logger.info(f"unable to find {item}")

# filter functions


def hasAbstract(data):
    """A filter function to get entries with abstracts"""
    return bool(data.article_data['abstract'] != "")
