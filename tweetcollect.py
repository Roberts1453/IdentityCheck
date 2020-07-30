from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from twitter_scraper import get_tweets
from tweetlib import tweetlib
from ibm_watson import PersonalityInsightsV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
import numpy as np
import datetime

##Watson authentication
apikey = 'alF3_IipD3RcJaDb4nxnFnJSBCTeqaWKQoR0Jow1Rqbb'
watsonurl = 'https://api.us-south.personality-insights.watson.cloud.ibm.com/instances/afdad2c4-68de-4894-9e95-797bcef306a3'
authenticator = IAMAuthenticator(apikey)
personality_insights = PersonalityInsightsV3(
    version='2017-10-13',
    authenticator=authenticator
)

personality_insights.set_service_url(watsonurl)

##Startup
print("Welcome to the Tweet Identity Check!")
print("Would you like to show progress updates during each step? y/n")
prog_in = input()
progress = False
if prog_in == 'y':
        progress = True


##collects recent tweets from the target user
print("Step: Get User Tweets")
print("Enter twitter username of target author: ")
user = input()
print("Enter date of last trustworthy tweet, following the format: " + str(datetime.date.today()) + "(or leave blank for today's date)")
lastdate = input()
if lastdate == '':
        lastdate = str(datetime.date.today())
print("Getting Tweets...")
ts = tweetlib.TweetLib(username=user, max_tweets=500, until=lastdate)
usertweets = ts.get_tweets()
usedwords = {}
realtweets = []
for tweet in usertweets:
	tweettext = tweet['text']
	realtweets.append(tweettext)
	for word in tweettext.split(' '):
		if word in usedwords:
			usedwords[word] += 1
		else:
			usedwords[word] = 1



#combines text of given tweets to one string to send to personality insights
def user_text(usertweets):
        request_input = ""
        for i in range(len(usertweets)):
                request_input += usertweets[i]
                if len(usertweets[i]) > 0:
                        last = usertweets[i][len(usertweets[i])-1]
                        if last != '.' and last != '!' and last != '?':
                                request_input += '.'
                request_input += ' '
        return request_input

#requests personality insights from IBM Watson on the author of given text
def get_personality(request_input):
        return personality_insights.profile(
                request_input,
                'application/json',
                content_type='text/plain',
                consumption_preferences=True,
                raw_scores=False
            ).get_result()

#returns number of words in given text
def word_count(text):
        words = text.split(' ')
        return len(words)


print("Step: Get Text")
#an array of bools identifying the author of the text as the target or not the target 
identity = []
#an array of strings containing at least 100 words for input to personality insights
requestinputtexts = []
currusertext = ''
for tweet in realtweets:
        currusertext += tweet
        if len(tweet) > 0:
                last = tweet[len(tweet)-1]
                if last != '.' and last != '!' and last != '?':
                        currusertext += '.'
        currusertext += ' '
        if word_count(currusertext) > 150:
                #print("Text length: " + str(word_count(currusertext)) + " words")
                requestinputtexts.append(currusertext)
                currusertext = ''
                identity.append(1)
#ensure the last text contains enough words for analysis to prevent error
while word_count(requestinputtexts[len(requestinputtexts) - 1]) < 110:
        requestinputtexts[len(requestinputtexts) - 1] += currusertext
print("Target User Text Count: " + str(len(requestinputtexts)))
                 
request_input = user_text(realtweets)


print("Step: Get Other Tweets")
##Get data
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
#Collects 500 English language tweets from the past day in North America
tweettotal = 500
ts = tweetlib.TweetLib(query_search="until:" + str(today) + " since:" + str(yesterday) + " lang:en geocode:39.8,-95.583068847656,2500km -filter:links -filter:replies", max_tweets=tweettotal)
datatweets = ts.get_tweets()
datatext = []
tweetcount = 0
#Collects additional tweets from each user to get enough text to analyze
for tweet in datatweets:
        tweetcount += 1
        if progress:
                print("Getting text #" + str(tweetcount) + " of " + str(tweettotal))
        curruser = tweet['username']
        ##old method. Error from too many requests
        #ts = tweetlib.TweetLib(username=curruser, max_tweets=50)
        #currtweets = ts.get_tweets()
        currtweets = list(get_tweets(curruser, pages=4))
        currtext = []
        for tweet in currtweets:
                if tweet['isRetweet'] is False:
                        currtext.append(tweet['text'])
        fulltext = user_text(currtext)
        while word_count(fulltext) < 150:
                fulltext += fulltext
        datatext.append(fulltext)
        requestinputtexts.append(fulltext)
        if curruser == user:
                identity.append(1)
        else:
                identity.append(0)


print("Step: Get Personality Data")
##Get array of personality data for each text
personalityData = []
personalitycount = 0
for text in requestinputtexts:
        personalitycount += 1
        if progress:
                print("Getting personality #" + str(personalitycount))
        profile = get_personality(text)
        values = []
        for elem in profile["values"]: #Big 5 personality gave less accurate test results, so values are used instead
                values.append(elem["percentile"])
        personalityData.append(values)

print("Step: Training model")
xTrain, xTest, yTrain, yTest = train_test_split(personalityData, identity, test_size = 0.2, random_state = None)
logreg=LogisticRegression()
logreg.fit(xTrain,yTrain)
accuracy = logreg.score(xTest, yTest)
print("Model ready!")
print("Model accuracy score: " + str(accuracy))
print("Input text to analyze (at least 100 words for greater accuracy): ")
inputtext = input()
while inputtext != '':
        while word_count(inputtext) < 120:
                inputtext += ". "
                inputtext += inputtext
        profile = get_personality(inputtext)
        big5 = [[]]
        for elem in profile["values"]:
                big5[0].append(elem["percentile"])
        #big5array = np.array(big5)
        #big5array.reshape(-1,1)
        print("Prediction: ")
        prediction = logreg.predict(big5)
        if prediction == 1:
                print("Author is target user.")
        else:
                print("Author is not target user.")
        print("Confidence in prediction: ")
        print(logreg.predict_proba(big5))
        print("Input next text to analyze (at leat 100 words): ")
        inputtext = input()

