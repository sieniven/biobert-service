import os
import sys
import json
import logging

from Publisher import initialize_twitter_api

sys.path.append("../")
from GTT.Model import Model
from GTT.Service import Service

class PublishService(Service):
    def __init__(self):
        Service.__init__(self)
        self.api = initialize_twitter_api()
        self.logger = logging.getLogger("GTT.Publish.Service")
        self.jobs = []

    def on_post(self, req, resp):
        data = json.loads(req.stream.read(req.content_length or 0))
        self.processDoc(data)
        
    def processDoc(self, data):
        input = Model()
        input.digest(data)
        input.add_to_history("publish")

        # get post results
        text = input.postData()
        print(text)
        # self.api.update_status(text)