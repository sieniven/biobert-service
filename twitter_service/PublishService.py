import os
import json
import logging

from rq import Queue
from redis import Redis

from Publisher import initialize_twitter_api 
from GTT.Model import Model
from GTT.Service import Service

class PublishService(Service):
    def __init__(self, RedisHost='localhost'):
        Service.__init__(self)
        self.api = initialize_twitter_api()
        self.logger = logging.getLogger("GTT.Publish.Service")
        self.q = Queue(connection=Redis(RedisHost))
        self.jobs = []

    def on_post(self, req, resp):
        data = json.loads(req.stream.read(req.content_length or 0))
        self.logger.debug(f"Attempting to post use handlers {str(self.posthandlers)}")
        
        self.jobs.append(self.q.enqueue(self.processDoc,
                                        on_success = self.report_success,
                                        on_failure = self.report_failure,
                                        kwargs={"data": data,
                                                "handlers": self.posthandlers}))

    def processDoc(self, data, handlers=[]):
        input = Model()
        input.digest(data)
        input.add_to_history("publish")
        self.api.update_status()