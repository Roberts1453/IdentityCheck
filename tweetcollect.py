from sklearn.linear_model import LinearRegression
#from twitter_scraper import Profile
from twitter_scraper import get_tweets
from ibm_watson import PersonalityInsightsV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
import numpy as np

##Watson authentication
apikey = 'alF3_IipD3RcJaDb4nxnFnJSBCTeqaWKQoR0Jow1Rqbb'
watsonurl = 'https://api.us-south.personality-insights.watson.cloud.ibm.com/instances/afdad2c4-68de-4894-9e95-797bcef306a3'
authenticator = IAMAuthenticator(apikey)
personality_insights = PersonalityInsightsV3(
    version='2017-10-13',
    authenticator=authenticator
)

personality_insights.set_service_url(watsonurl)


##collects recent tweets from the target user
user = 'twitter'
tweets = list(get_tweets(user, pages=5))
usedwords = {}
realtweets = []
for tweet in tweets:
	if tweet['isRetweet'] is False:
		tweettext = tweet['text']
		print(tweettext)
		realtweets.append(tweettext)
		for word in tweettext.split(' '):
			if word in usedwords:
				usedwords[word] += 1
			else:
				usedwords[word] = 1




##requests personality insights from IBM Watson on the target user
request_input = ""
for i in range(10):
        request_input += realtweets[i]
        last = realtweets[i][len(realtweets[i])-1]
        if last != '.' and last != '!' and last != '?':
                request_input += '.'
        request_input += ' '
        
def get_personality(request_input):
        return personality_insights.profile(
                request_input,
                'application/json',
                content_type='text/plain',
                consumption_preferences=True,
                raw_scores=False
            ).get_result()

profile = get_personality(request_input)

big5 = []
for elem in profile["personality"]:
	big5.append(elem["percentile"])
print(big5)
	
'''
#an array of arrays of personality data
personalityData = []
#an array of bools identifying the author of the text as the target or not the target 
identity = []
linreg=LinearRegression()
linreg.fit(personalityData,identity)

'''
