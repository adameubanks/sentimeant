from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords, wordnet
from nltk import download
from io import open
import numpy as np
import json
import re

download('stopwords')
download('wordnet')

#Get rid of noise from dataset
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    word_list = string.split(' ')
    string = ""
    for word in word_list:
        if word not in stopwords.words('english'):
            if wordnet.synsets(word):
                string = string + word + " "
    return string.strip().lower()

#For turning long text into sentences to put into each model
def sequence_text (text, max_len):
    word_list = text.split(' ')
    filtered_word_list = word_list[:]
    for word in word_list:
        if word in stopwords.words('english'):
            filtered_word_list.remove(word)
    word_list = filtered_word_list
    sentences = []
    i = 0
    while i < len(word_list):
        k = 0
        sentence = ""
        while k < max_len:
            if k+i >= len(word_list):
                pass
            else:
                sentence=sentence+word_list[i+k]+" "
            k+=1
        sentences.append(sentence)
        i+=max_len

    return sentences


#Political Bias Stuff - modified from https://github.com/icoen/CS230P/blob/master/RNN/data_helpers2.py
def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    positive_labels = [[1] for _ in positive_examples]
    negative_labels = [[0] for _ in negative_examples]

    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

#returns tokenized data
def preprocessPoliticalData(dem_file,rep_file, vocab_size):
    print("Loading data...")
    x_text, y = load_data_and_labels(dem_file,rep_file)
    tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", oov_token="<OOV>")
    tokenizer.fit_on_texts(x_text)
    x_train = tokenizer.texts_to_sequences(x_text)

    tokenizer_json = tokenizer.to_json()
    with open('politicaltokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    x = np.array(list(pad_sequences(x_train, 50, padding='post', truncating='post')))

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_train = x[shuffle_indices]
    y_train = y[shuffle_indices]

    del x, y

    return x_train, y_train
