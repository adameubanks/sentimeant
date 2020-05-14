from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from numpy import abs
# import urllib.request
import re
# from bs4 import BeautifulSoup
from nltk import download, tokenize
# from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model
import pickle

download('vader_lexicon')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    url = request.form['url']
    if url != '':
        text = "url"
        # soup = BeautifulSoup(urllib.request.urlopen(url).read(),"lxml")
        # html = soup.get_text().split(" ")
        # text = ""
        # download('wordnet')
        # for word in html:
        #     if wordnet.synsets(word):
        #         text = text + word + " "
    else:
        text = request.form['text']
    print(text)

    data = pd.DataFrame(data=[text],columns=["Text"])
    sentiment = SentimentIntensityAnalyzer()
    #sentiment analysis
    all_sentiments = []
    for index, row in data.iterrows():
      ss = sentiment.polarity_scores(row['Text'])
      all_sentiments.append(ss)

    negative=neutral=positive=compound = 0
    for i in range(len(all_sentiments)):
      negative += all_sentiments[i]['neg']
      neutral += all_sentiments[i]['neu']
      positive += all_sentiments[i]['pos']
      compound += all_sentiments[i]['compound']

    #political bias
    words = text.split(' ')
    sentences = []
    while i < len(words):
        k = 0
        sentence = ""
        while k < 65:
            if k+i >= len(words):
                pass
            else:
                sentence=sentence+words[i+k]+" "
            k+=1
        sentences.append(sentence)
        i+=65

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    int2label = {
        0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'
    }
    print(len(sentences))

    # feeling_model = load_model('feeling_model.h5')
    political_bias = load_model('political_bias.h5')
    conservative=liberal = 0
    anger=fear=joy=love=sadness=surprise = 0
    for i in sentences:
        sentence = tokenizer.texts_to_sequences([i])
        sentence = pad_sequences(sentence, 65, padding='post', truncating='post')
        p_score = political_bias.predict(sentence)
        liberal+=p_score
        conservative+=(1-p_score)

        # f_score = feeling_model.predict(sentence)
        # anger+=f_score[0][0]
        # fear+=f_score[0][1]
        # joy+=f_score[0][2]
        # love+=f_score[0][3]
        # sadness+=f_score[0][4]
        # surprise+=f_score[0][5]


    conservative /= len(sentences)
    liberal /= len(sentences)
    liberal = liberal[0][0]
    conservative = conservative[0][0]

    anger /= len(sentences)
    fear /= len(sentences)
    joy /= len(sentences)
    love /= len(sentences)
    sadness /= len(sentences)
    surprise /= len(sentences)

    return render_template('results.html', text=text, url=url, positive=positive, negative=negative, neutral=neutral, compound=compound,
    conservative=conservative, liberal=liberal*100,
    anger=round(anger*100,1), fear=round(fear*100,1), joy=round(joy*100,1), love=round(love*100,1), sadness=round(sadness*100,1), surprise=round(surprise*100, 1))


if __name__ == "__main__":
    app.run(debug=True)
