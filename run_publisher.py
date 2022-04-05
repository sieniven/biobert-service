import sys
import falcon
import logging
import waitress

from GTT.Service import Service
from PublishService import PublishService

def main():
    app = PublishService()
    logger = logging.getLogger(__name__)

    for h in app.posthandlers:
        logger.info(f"Attached handler: {h}")

    api = falcon.App()
    api.add_route('/', app)
    waitress.serve(api, host='127.0.0.1', port=8010)

if (__name__ == "__main__"):
    # run biobert as a process
    main()