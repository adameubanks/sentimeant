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
    return render_template('home.html')

@app.route('/getstarted')
def form():
    return render_template('payment.html',pub_key=pub_key)

@app.route('/diagnostic',methods=["POST"])
def diagnostic():
    customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])
    charge = stripe.Charge.create(customer=customer.id, amount=500, currency='usd',description='sentiment analysis')
    username = request.form.get('username', type=str)

    consumer_key = 'oHE7XfwvO3TekFfoqEboGn1tv'
    consumer_secret = 'Q77xcmzuA2qv2CSOHvFIBVDoAIGbVOWlEYfSQ7v9rERmsqPic4'
    access_token = '2586611642-FTYXWePK0mNzMhJZIQBzArMuSli9WVXkBZrkb24'
    access_token_secret = '70sJYhc3H86FQY1IR9TpoPsm99fDlbAnBXKNE5eNe2ooX'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    data = pd.DataFrame(data=[tweet.text for tweet in
                              tweepy.Cursor(api.search, q=username, result_type="recent",
                                            include_entities=True, lang="en").items(200)], columns=['Tweets'])
    import nltk
    nltk.download('vader_lexicon')
    sentiment = SentimentIntensityAnalyzer()

    all_sentiments = []
    for index, row in data.iterrows():
        ss = sentiment.polarity_scores(row['Tweets'])
        all_sentiments.append(ss)

    negative = neutral = positive = compound = 0
    print(len(all_sentiments))
    for i in range(len(all_sentiments)):
        negative += all_sentiments[i]['neg']
        neutral += all_sentiments[i]['neu']
        positive += all_sentiments[i]['pos']
        compound += all_sentiments[i]['compound']

    return render_template('diagnostic.html', username=username, positive=positive, negative=negative, neutral=neutral, compound=compound)

if __name__ == "__main__":
    app.run(debug=True)
