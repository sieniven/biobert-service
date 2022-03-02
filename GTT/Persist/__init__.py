"""An attempt to persist article information in a single Mongo collection"""

from GTT.config import get_config
from pymongo import MongoClient
from GTT.SetupLogging import setup_logging

logger = setup_logging()

try:
    client = MongoClient(host=get_config("mongo_server"),
                         port=get_config("mongo_port"))
    db_name = get_config("mongo_db")
    client[db_name].authenticate(get_config("mongo_user"),
                                 get_config("mongo_pwd"),
                                 mechanism='SCRAM-SHA-1')
    db = client[db_name]
except Exception as e:
    logger.warning(f"could not authenticate into Mongo: {e}")
    client = None
    db = None
