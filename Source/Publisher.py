import tweepy
import yaml

def initialize_twitter_api():
    with open('../twitter_config.yaml', "r") as f:
        config = yaml.safe_load(f)

    api_key = config["consumer_api_key"]
    api_key_secret = config["consumer_api_key_secret"]
    access_token = config["access_token"]
    access_token_secret = config["access_token_secret"]

    # authenticate to Twitter
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    try:
        api.verify_credentials()
        print("Authetication successful.")
    except:
        print("Authentication unsuccessful.")

    return api
