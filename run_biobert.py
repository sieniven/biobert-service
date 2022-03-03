import sys
import falcon
import logging
import waitress

from BiobertService import BiobertService

def main():
    app = BiobertService()
    logger = logging.getLogger(__name__)

    for h in app.posthandlers:
        logger.info(f"Attached handler: {h}")

    api = falcon.App()
    api.add_route('/', app)
    waitress.serve(api,host='0.0.0.0',port=10000)

def processDataset():
    app = BiobertService()
    app.process_ingest_dataset()

if (__name__ == "__main__"):
    # run biobert as a process
    # main()
    processDataset()