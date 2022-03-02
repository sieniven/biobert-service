import logging                              # to log module processes

from GTT.Ingest.PubMedArticle import PubMedArticle
from GTT.utils import chunks
from GTT.Service import Service
from Bio import Entrez
from datetime import datetime, timedelta
import inspect
from email.utils import parseaddr

class PubMed(Service):
    logger = logging.getLogger(__name__)
    email = ""
    api_key = None
    retmax = 1000

    def __init__(self,PMIDs=[]):
        """A collection of PubMedArticle records """

        Service.__init__(self)
        self.articles = []
        self.logger = logging.getLogger(__name__)
        self.ingest(PMIDs)

    def ingest(self,ids):
        """ingest a collection of PubMedArticles
        @args
          ids: a list of ids, from a search or from some other input (e.g., gene2pubmed)

        Reading from Entrez.efetch occasionally fails, often because of a single article. We don't
        know where the problem is, so in such a case this function works recursively on chunks of the input
        """
        ids = list(set(ids))
        if (len(ids) > 0):
            recurrent = (inspect.stack()[1].function == inspect.stack()[0].function)
            if (not recurrent):
                self.logger.info("Attempting to ingest {} items from PubMed".format(len(ids)))
            else:
                self.logger.info("Ingestion batch size now {}".format(len(ids)))
            handle = Entrez.efetch(db='pubmed',id=ids,retmode='xml',retmax=self.__class__.retmax)
            try:
                records = Entrez.read(handle)
                records = records['PubmedArticle'] + records['PubmedBookArticle']
                self.logger.info("Identified {} records".format(len(records)))
                for record in records:
                    self.articles.append(PubMedArticle(record))
            except NotImplementedError:
                batchSize = max(round(len(ids) / 4),1)
                if (isinstance(ids,list) and (len(ids) > 1)):
                    for chunk in chunks(ids,batchSize):
                        self.ingest(chunk)
                else:
                    pass
            if (not recurrent):
                self.logger.info("Successfully ingested {} items from PubMed".format(len(self.articles)))

    @classmethod
    def config(cls,email,api_key=None):
        cls.email = parseaddr(email)[1]
        Entrez.email = cls.email
        if (api_key):
            cls.api_key = api_key
            Entrez.api_key = cls.api_key

    @classmethod
    def search(cls,query):
        """get IDs matching a search"""
        try:
            handle = Entrez.esearch(db='pubmed',
                                    retmax=cls.retmax,
                                    retmode='xml',
                                    term=query)
            results = Entrez.read(handle)
            return results['IdList']
        except Exception as e:
            cls.logger.error(f"Entrez search failed with query {query}: {e}")

    @classmethod
    def dateToQuery(cls,startDate=((datetime.utcnow() - timedelta(days=2)).strftime("%Y/%m/%d")),
                    endDate=((datetime.utcnow() - timedelta(days=2)).strftime("%Y/%m/%d"))):
        """ Formats user-specified dates into an appropriate query string (date interval) to be used for searching Pubmed
            @args
                startDate (str): Start of date interval from which Pubmed papers are to be ingested
                    default is one whole day, the day before the function is run.
                endDate   (str): End of date interval from which Pubmed papers are to be ingested
                * Note: Specified dates must be in this format: "%Y/%m/%d"
            @returns
                string (date interval in a format recognised by Pubmed API as a query)
        """
        query = str(startDate) + ":" + str(endDate) + "[edat]"
        return query
