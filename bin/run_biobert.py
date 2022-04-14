import sys
import falcon
import logging
import waitress

sys.path.append("../")
from GTT.Service import Service

sys.path.append("../Source/")
from BiobertService import BiobertService

def main():
    publishHandler = Service.makeRequestHandler("http://localhost:8010")
    app = BiobertService(publishHandler)
    logger = logging.getLogger(__name__)

    api = falcon.App()
    api.add_route('/', app)
    waitress.serve(api, host='127.0.0.1', port=10000)

def processDataset():
    publishHandler = Service.makeRequestHandler("http://localhost:8010")
    app = BiobertService(publishHandler)
    app.process_ingest_dataset()


if (__name__ == "__main__"):
    # run biobert as a process
    # main()
    processDataset()