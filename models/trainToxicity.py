''' This code was run in a kaggle notebook written by me with TPU acceleration. You
can view the full notebook here: https://www.kaggle.com/khanradcoder/simple-toxicity-classification-with-gru '''

from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Embedding, GRU, GlobalMaxPooling1D, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
import tensorflow as tf
import pandas as pd
import numpy as np
import json

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

dataset = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
x = dataset["comment_text"]
y = dataset["target"]
max_len = 40
vocab = 50000

def tokenizeToxicity(file, vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", oov_token="<OOV>")
    tokenizer.fit_on_texts(file)
    tokenized_file = tokenizer.texts_to_sequences(file)
    tokenizer_json = tokenizer.to_json()
    with open('toxictokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    tokenized_padded_file = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(tokenized_file, max_len, padding='post', truncating='post')))
    return tokenized_padded_file

x_tokenized = tokenizeToxicity(x, vocab)
x_train, x_test, y_train, y_test = train_test_split(x_tokenized, y, test_size=0.2, random_state=42)


#Build model
def biGRUnet():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(vocab,max_len,input_length=max_len)(inputs)
    layer = Bidirectional(GRU(64, return_sequences = True), merge_mode='concat')(layer)
    layer = GlobalMaxPooling1D()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

with tpu_strategy.scope():
    model = biGRUnet()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=64,epochs=3,validation_data=(x_test, y_test))
model.save("toxicity.h5")
