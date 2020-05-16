'''Code, data and preprocessing originally from a Stanford NLP project. See the full code here:
https://github.com/icoen/CS230P
and the paper:
https://cs230.stanford.edu/projects_fall_2018/reports/12449209.pdf
Special thanks to Eddie Sun who helped me out with interpreting the code'''

from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Embedding, GRU, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
import preprocessing
import numpy as np
import json

#Build model
def biGRUnet():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(vocab_size,50,input_length=max_len)(inputs)
    layer = Bidirectional(GRU(64, return_sequences = True), merge_mode='concat')(layer)
    layer = GlobalMaxPooling1D()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

vocab_size = 50000

#Preprocess data
x_train, y_train = preprocessing.preprocessPoliticalData('data/political_bias/demfulltrain.txt', 'data/political_bias/repfulltrain.txt', vocab_size)

max_len = x_train.shape[1]

model = biGRUnet()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#Get validation data
x_vtext, y_val = preprocessing.load_data_and_labels('data/political_bias/demfullval.txt', 'data/political_bias/repfullval.txt')

#Get tokenizer from preprocessing file
with open('politicaltokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

x_val = np.array(list(pad_sequences(tokenizer.texts_to_sequences(x_vtext), max_len, padding='post', truncating='post')))

#Train and save model
model.fit(x_train,y_train,batch_size=64,epochs=5,
          validation_data=(x_val, y_val))
model.save('political_bias.h5')
