from flask import Flask, render_template, request, redirect, url_for
import stripe
import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

pub_key = "pk_test_ZTlu8ZeGhxcvr013B9tXDcji008aGb26fG"
sec_key = "sk_test_NISIDuFJvjhYaRrSXF9Yn3gd006rKr2Xd8"

stripe.api_key = sec_key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    return render_template('search.html',pub_key=pub_key)

@app.route('/profile')
def profile():
    # customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])
    # charge = stripe.Charge.create(customer=customer.id, amount=500, currency='usd',description='sentiment analysis')
    search_term = request.args['search_term']

    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    # data = pd.DataFrame(data=[tweet.text for tweet in
    #                           tweepy.Cursor(api.search, q=search_term, result_type="recent",
    #                                         include_entities=True, lang="en").items(200)], columns=['Tweets'])
    # import nltk
    # nltk.download('vader_lexicon')
    # sentiment = SentimentIntensityAnalyzer()
    #
    # all_sentiments = []
    # for index, row in data.iterrows():
    #     ss = sentiment.polarity_scores(row['Tweets'])
    #     all_sentiments.append(ss)
    #
    # negative = neutral = positive = compound = 0
    # print(len(all_sentiments))
    # for i in range(len(all_sentiments)):
    #     negative += all_sentiments[i]['neg']
    #     neutral += all_sentiments[i]['neu']
    #     positive += all_sentiments[i]['pos']
    #     compound += all_sentiments[i]['compound']
    user = api.get_user(search_term)

    worse_photo_url = user.profile_image_url
    better_photo_url = user.profile_image_url[:len(user.profile_image_url)-10]+"400x400.jpg"
    compound = 1
    negative = 2
    neutral = 3
    positive = 4
    liberal = 0
    conservative = 0
    return render_template('profile.html', photo_url=better_photo_url, search_term=search_term, positive=positive,
     negative=negative, neutral=neutral, compound=compound, liberal=liberal, conservative=conservative)

if __name__ == "__main__":
    app.run(debug=True)
