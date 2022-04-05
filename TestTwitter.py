import tweepy
from Publisher import initialize_twitter_api, initialize_twitter

api = initialize_twitter_api()
text = "hello!"

api.update_status(status=text)