

import logging                              # to log module processes

from GTT.Ingest.bioRxivPreprint import bioRxivPreprint
import requests
from GTT.Service import Service
from datetime import datetime, timedelta

api_url = 'https://api.biorxiv.org/details/biorxiv/'


class bioRxiv(Service):

    def __init__(self,daterange=None):

        """A collection of bioRxivPreprint records """

        Service.__init__(self)
        self.articles = []
        self.logger = logging.getLogger(__name__)

        if (daterange):
            self.ingest(daterange)

    def ingest(self,daterange=None):
        """ingest a collection of bioRxivPreprints
        @args
           daterange: A string: two dates forward slash separated '2021-07-26/2021-07-26'
        """
        if (daterange):
            cursor = 0
            total = 100
            results = []
            try:
                while cursor < total:
                    url = "{api_url}/{daterange}/{cursor}/json".format(api_url=api_url,daterange=daterange,cursor=cursor)
                    response = requests.get(url)
                    data = response.json()
                    total = data['messages'][0]['total']
                    results = results + data['collection']
                    cursor = len(results)
                preprints = [bioRxivPreprint(a) for a in results]
                seen = set()
                self.articles = [a for a in preprints
                                 if a.id not in seen
                                 and not seen.add(a.id)]
            except Exception as e:
                self.logger.error("We've got an ingestion problem in here: {}".format(e))

    @classmethod
    def dateToQuery(cls,startDate=(datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d"),
                    endDate=(datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")):
        """ Formats user-specified dates into an appropriate query string (date interval) to be used for searching Pubmed
            @args
                startDate (str): Start of date interval from which preprints are to be ingested
                    default is one whole day, two days before the function is run (to ensure it is complete in every time zone).
                endDate   (str): End of date interval from which Pubmed papers are to be ingested
                * Note: Specified dates must be in this format: "%Y-%m-%d"
            @returns
                string (date interval in a format recognised by bioRxiv REST API as a query)
        """
        return "{}/{}".format(str(startDate),str(endDate))
