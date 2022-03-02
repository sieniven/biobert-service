"""A singleton of a mongodb collection"""

from GTT.config import get_config
from GTT.SetupLogging import setup_logging
from GTT.Persist import db

logger = setup_logging()

if db:
    try:
        coll = db[get_config("mongo_article_collection")]
    except Exception as e:
        logger.coll(f"could not use collection {get_config('mongo_collection')}: {e}")
        coll = None
else:
    coll = None
