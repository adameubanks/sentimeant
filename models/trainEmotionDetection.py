'''Code and data modified from Manuel Romero. See the original colab notebook here:
https://colab.research.google.com/drive/1KPd6e6YkTPJQkNRhvYsZCM6YJZ8RkiIu
'''

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import preprocessing
import numpy as np
import pickle
import json
import re

#Get data
def load_from_pickle(directory):
    return pickle.load(open(directory,"rb"))

data = load_from_pickle(directory="data/emotion/emotion_dataset.pkl")
print("Data Loaded")

emotions = [ "sadness", "joy", "love", "anger", "fear", "surprise"]
data = data[data["emotions"].isin(emotions)]

y = data['emotions'].values
comments_train = data['text']
comments_train = list(comments_train)
max_text = (max(comments_train, key=len))

def num_words(sentence):
  words = sentence.split()
  return len(words)
total_avg_words = sum( map(num_words, comments_train) ) / len(comments_train)
max_len = 50
vocab_size = 10000
embedding_dim = 100

#Clean text
texts = []
for line in comments_train:
    texts.append(preprocessing.clean_str(line))
print("Cleaned data")

#Tokenize text
tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, padding = 'post', maxlen = max_len)
print("Tokenized data")

#Save tokenizer
tokenizer_json = tokenizer.to_json()
with open('emotokenizer.json', 'w', encoding='utf-8') as f:
  f.write(json.dumps(tokenizer_json, ensure_ascii=False))

#Shuffle indices
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = y[indices]

#Onehot encode label data
lb = LabelBinarizer()
lb.fit(labels)
lb.classes_
labels = lb.transform(labels)

#Split train and validation set
num_validation_samples = int(0.2*data.shape[0])
x_train = data[: -num_validation_samples]
y_train = labels[: -num_validation_samples]
x_val = data[-num_validation_samples: ]
y_val = labels[-num_validation_samples: ]

#Split validation and test set
x_val = x_val[: -40000]
y_val = y_val[: -40000]
x_test = x_val[-40000: ]
y_test = y_val[-40000: ]

#Use glove embedding
embeddings_index = {}
f = open('data/emotion/glove.6B.100d.txt')
print('Loading GloVe')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n Proceeding with Embedding Matrix...", end="")

embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


#Build the model
sequence_input = Input(shape=(max_len,), dtype='int32')
embedding_layer = Embedding(len(word_index) + 1,
                           embedding_dim,
                           weights = [embedding_matrix],
                           input_length = max_len,
                           trainable=False,
                           name = 'embeddings')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
preds = Dense(6, activation="softmax")(x)

model = Model(sequence_input, preds)
model.compile(loss = 'categorical_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])

print('Training progress:')
model.fit(x_train, y_train, epochs = 10, batch_size=128, validation_data=(x_val, y_val))

print("Accuracy in the test set:")
model.evaluate(x_test, y_test)[1]

model.save('feeling_model.h5')
