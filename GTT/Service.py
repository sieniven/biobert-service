import logging
from GTT.Model import Model
import requests
import validators


class Service():

    def __init__(self):
        self.posthandlers = []
        self.articles = [Model()]
        self.logger = logging.getLogger(__name__)

    def postData(self):
        """Child classes may override the postData() method"""
        data = [a.postData() for a in self.articles]
        return list(filter(None.__ne__,data))

    def filter(self,filterfun):
        """Run a filter on the articles and update the results

        @args
          filterfun: must take one argument, an article (GTT.Model). Must return a boolean"
        """
        self.logger.debug("Filtering using {str(filterfun)} ")
        try:
            articles = list(filter(filterfun,self.articles))
            self.articles = articles
            self.logger.debug(f"Filtered collection contains {len(self.articles)} articles after {filterfun}")
        except Exception as E:
            self.logger.error(f"Filter function {str(filterfun)} failed: {E}")

    def addPostHandler(self,postfn):
        """Posthandler functions will be used to post to the next service from an instance of this class

        @args:
           postfn: a handler function taking one argument, expected to be the output of self.postData()
        """

        self.posthandlers.append(postfn)
        self.logger.info(f"Added handler {str(postfn)} to {str(self)}")

    def addEvent(self,event):
        """Add an event to the objects being processed by the service"""
        for article in self.articles:
            article.add_to_history(event)

    def post_many(self,handlers=[]):
        for posthandler in filter(callable,(self.posthandlers + handlers)):
            n_posted = 0
            for article in self.articles:
                try:
                    posthandler(article)
                    n_posted = n_posted + 1
                except Exception as E:
                    self.logger.error(f"Failed to post {str(article)} using {str(posthandler)}: {E}")
            self.logger.info(f"posted {n_posted} {type(self).__name__} records using {str(posthandler)}")

    def __iter__(self):
        return ServiceIterator(self)

    @classmethod
    def makeRequestHandler(cls,url):
        """Create a new handler to post requests from service to service

        @args
           url: logged if url is invalid, which might be ok on Docker microservices

        @returns
           a requestHandler function that can be used for the post_many(self) method of the Service class
    """
        logger = logging.getLogger(__name__)

        if (validators.url(url) is not True):
            logger.debug(validators.url(url))

        def requestHandler(data):
            try:
                if isinstance(data,Model):
                    data = data.postData()
                r = requests.post(url=url, json=data)
            except Exception as e:
                logging.error(f"Posting failed! Error: {e}")

        return requestHandler


class ServiceIterator():
    def __init__(self,ServiceCollection):
        self._articles = ServiceCollection.articles
        self._index = 0
        self._size = len(self._articles)

    def __next__(self):
        """Returns the next value"""
        if (self._index < self._size):
            article = self._articles[self._index]
            self._index = self._index + 1
            return article
        raise StopIteration
