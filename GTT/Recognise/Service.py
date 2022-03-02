# import requests
from redis import Redis
from rq import Queue
import logging
import json
from pprint import pprint
from GTT.Recognise.Document import Document
from GTT.Service import Service

logger = logging.getLogger(__name__)


class RecogniseService(Service):

    fileDumpLocation = '/tmp/recognitionDump.json'

    def __init__(self,RedisHost='localhost'):
        Service.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.jobs = []
        self.q = Queue(connection=Redis(RedisHost))

    def on_post(self,req,resp):

        data = json.loads(req.stream.read(req.content_length or 0))
        self.logger.debug(f"attempting to post use handlers {str(self.posthandlers)}")
        self.jobs.append(self.q.enqueue(processDoc,
                                        on_success=report_success,
                                        on_failure=report_failure,
                                        kwargs={"data": data,
                                                "handlers": self.posthandlers}))

def processDoc(data,handlers=[]):
    logger = logging.getLogger("GTT.Recognise.Service")
    doc = Document(data)
    id = doc.id
    n = len(doc.entities)
    logger.info(f"NER finished for {id} with {n} entities recognised")
    doc.update() 
    logger.info(f"{id} updated")
    doc.post_one(handlers)
    return doc


def report_success(job, connection, result, *args, **kwargs):
    pass


def report_failure(job, connection, type, value, traceback):
    logging.getLogger(__name__).error(traceback)
