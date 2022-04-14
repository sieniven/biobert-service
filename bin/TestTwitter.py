import sys
import tweepy

sys.path.append("../Source/")
from Publisher import initialize_twitter_api, initialize_twitter

api = initialize_twitter_api()
text = "hello!"

api.update_status(status=text)