# from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from flask import Flask, render_template, request, redirect, url_for
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from models import preprocessing
import tensorflow as tf
# from nltk import download
import pandas as pd
import numpy as np
import json
import re

# download('vader_lexicon')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    url = request.form['url']
    text = request.form['text']

    max_len = 50
    cleaned_text = preprocessing.clean_str(text)
    sentences = preprocessing.sequence_text(cleaned_text, max_len)

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
    political_bias = load_model('models/political_bias.h5')
    with open('models/politicaltokenizer.json') as f:
        data = json.load(f)
        political_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    political_tokenized = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(political_tokenizer.texts_to_sequences(sentences), max_len, padding='post', truncating='post')))


    #emotion
    emotion_model = load_model('models/feeling_model.h5')
    with open('models/emotokenizer.json') as f:
        data = json.load(f)
        emotion_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    emotion_tokenized = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(emotion_tokenizer.texts_to_sequences(sentences), max_len, padding='post', truncating='post')))

    liberal = political_bias.predict(political_tokenized)[0][0]
    conservative = 1-liberal

    f_score = emotion_model.predict(emotion_tokenized)
    anger=f_score[0][0]
    fear=f_score[0][1]
    joy=f_score[0][2]
    love=f_score[0][3]
    sadness=f_score[0][4]
    surprise=f_score[0][5]


    return render_template('results.html', text=text, url=url, positive=positive, negative=negative, neutral=neutral, compound=compound,
                            conservative=conservative, liberal=liberal*100, anger=round(anger*100,1), fear=round(fear*100,1), joy=round(joy*100,1),
                             love=round(love*100,1), sadness=round(sadness*100,1), surprise=round(surprise*100,1))


if __name__ == "__main__":
    app.run(debug=True)
