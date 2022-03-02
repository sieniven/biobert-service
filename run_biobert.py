import falcon
import logging
import waitress

from BiobertService import BiobertService
from BiobertProcessDataset import process_ingest_dataset

def main():
    app = BiobertService()
    logger = logging.getLogger(__name__)

    for h in app.posthandlers:
        logger.info(f"Attached handler: {h}")

    api = falcon.App()
    api.add_route('/', app)
    waitress.serve(api,host='0.0.0.0',port=10000)

if (__name__ == "__main__"):
    # run biobert as a process
    main()

    # run biobert to process ingest dataset
    # process_ingest_dataset()