import json
import logging

from rq import Queue
from redis import Redis

from Biobert import Biobert
from BiobertModel import BiobertModel
from GTT.Service import Service

class BiobertService(Service):
    def __init__(self, RedisHost='localhost'):
        self.logger = logging.getLogger("GTT.Biobert.Service")
        self.biobert = Biobert()
        self.jobs = []
        self.q = Queue(connection=Redis(RedisHost))

    def on_post(self, req, resp):
        data = json.loads(req.stream.read(req.content_length or 0))
        self.logger.debug(f"attempting to post use handlers {str(self.posthandlers)}")
        
        self.jobs.append(self.q.enqueue(self.processDoc,
                                        on_success = self.report_success,
                                        on_failure = self.report_failure,
                                        kwargs={"data": data,
                                                "handlers": self.posthandlers}))

    def processDoc(self, data, handlers=[]):
        doc = BiobertModel(data)
        self.biobert.load_data(doc)

        # run prediction for input data
        self.biobert.predict()

        # process data
        # TBC

        id = doc.id
        n = len(doc.entities)
        self.logger.info(f"NER finished for {id} with {n} entities recognized with Biobert")       
        doc.post_one(handlers)
        
        return doc

    def report_success(self, job, connection, result, *args, **kwargs):
        pass

    def report_failure(self, job, connection, type, value, traceback):
        self.logger.error(traceback)
