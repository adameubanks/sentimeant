# from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from flask import Flask, render_template, request, redirect, url_for
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from models import preprocessing
import tensorflow as tf
from nltk import download
import pandas as pd
import numpy as np
import json
import re
import urllib.request

# download('vader_lexicon')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/howitworks")
def howitworks():
    return render_template("howitworks.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results', methods=['POST'])
def results():
    url = '' #request.form['url']
    if url != "":
        with urllib.request.urlopen(url) as address:
            text = address.read().decode('utf-8')
        text = re.sub(r'<.*?>', '', text)
    else:
        text = request.form['text']
    # url = request.form['url']

    max_len = 50
    text = preprocessing.clean_str(text)
    sentences = preprocessing.sequence_text(text, max_len)


    #get model type
    model = request.form['model_type']


    #set all params to 0
    negative=neutral=positive=compound=liberal=conservative=anger=fear=joy=love=sadness=surprise=toxicity=0

    #sentiment analysis
    if model == 'polarity':
        data = pd.DataFrame(data=[text],columns=["Text"])
        sentiment = SentimentIntensityAnalyzer()

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
    if model == 'political':
        political_bias = load_model('models/political_bias.h5')
        with open('models/politicaltokenizer.json') as f:
            data = json.load(f)
            political_tokenizer = preprocessing.tokenizer_from_json(data)
        political_tokenized = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(political_tokenizer.texts_to_sequences(sentences), max_len, padding='post', truncating='post')))
        liberal = political_bias.predict(political_tokenized)[0][0]
        conservative = 1-liberal

    #emotion
    if model == 'emotion':
        emotion_model = load_model('models/feeling_model.h5')
        with open('models/emotokenizer.json') as f:
            data = json.load(f)
            emotion_tokenizer = preprocessing.tokenizer_from_json(data)
        emotion_tokenized = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(emotion_tokenizer.texts_to_sequences(sentences), max_len, padding='post', truncating='post')))
        f_score = emotion_model.predict(emotion_tokenized)
        anger=f_score[0][0]
        fear=f_score[0][1]
        joy=f_score[0][2]
        love=f_score[0][3]
        sadness=f_score[0][4]
        surprise=f_score[0][5]

    #toxicity
    if model == 'toxicity':
        toxicity = load_model('models/toxicity.h5')
        with open('models/toxictokenizer.json') as f:
            data = json.load(f)
            toxicity_tokenizer = preprocessing.tokenizer_from_json(data)
        toxicity_tokenized = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(toxicity_tokenizer.texts_to_sequences(sentences), max_len, padding='post', truncating='post')))
        toxicity = toxicity.predict(toxicity_tokenized)[0][0]

    return render_template('results.html', model=model, text=text, url=url, positive=positive, negative=negative, neutral=neutral, compound=compound,
                            conservative=conservative*100, liberal=liberal*100, anger=round(anger*100,1), fear=round(fear*100,1), joy=round(joy*100,1),
                             love=round(love*100,1), sadness=round(sadness*100,1), surprise=round(surprise*100,1), toxicity=round(toxicity*100,1))




if __name__ == "__main__":
    app.run(debug=True)
