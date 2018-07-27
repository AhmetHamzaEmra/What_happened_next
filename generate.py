from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import pickle
import random
from tqdm import tqdm
import numpy as np
import nltk 
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import Embedding


!!! Dont forget to change the name of the script !!!
with open("script.txt", 'r') as f:
    text = f.read()

# Little bit of text cleaning 
text = text.replace("\n", " ")
text = text.replace("          ", " ")
text = text.lower()

# Word tokenization
words_t = nltk.tokenize.word_tokenize(text)
words = list(set(words_t))
print("Number of words: ", len(words))

# Create maping w2i and i2w
# Check if we already create the mappings 
word2id_file = 'word2id.pickle'
word_indices = None
if os.path.isfile(word2id_file):
    print('Loading previous word2id')
    word_indices = pickle.load(open(word2id_file, 'rb'))
else:
    word_indices = dict((c, i) for i, c in enumerate(words))
    pickle.dump(word_indices, open(word2id_file,'wb'))

id2word_file = 'id2word.pickle'
indices_word = None
if os.path.isfile(id2word_file):
    print('Loading previous id2word')
    indices_word = pickle.load(open(id2word_file, 'rb'))
else:
    indices_word = dict((i, c) for i, c in enumerate(words))
    pickle.dump(indices_word, open(id2word_file,'wb'))




# Sequence dataset 
X = []
Y = []
seq_len = 50
for w in tqdm(range(len(words_t[:-seq_len]))):
    seq = words_t[w:w+seq_len]
    seq_v = []
    for i in seq:
        seq_v.append(word_indices[i])
    X.append(seq_v)
    a_ = np.zeros([len(words)])
    a_[word_indices[words_t[w+seq_len]]]=1
    Y.append(a_)
X = np.array(X)
Y = np.array(Y)


# Our model
model = Sequential()
model.add(Embedding(len(words), 512))
model.add(LSTM(128, input_shape=(50, 512), activation="tanh", return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, activation="tanh"))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(len(words), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.summary()

model.fit(X, Y, epochs=100, batch_size=64, callbacks=callbacks_list)


# generate 500 words
start = words_t[-seq_len:]
seed = []
for i in start:
    seed.append(word_indices[i])
generated_text = seed
print("Seed:")
print(seed)
seed = np.array(seed)
seed = seed.reshape(1,seed.shape[0])
for i in tqdm(range(500)):
    
    next_word_index = np.argmax(model.predict(seed))
    
    generated_text.append(next_word_index)
    seed = generated_text[-seq_len:]
    seed = np.array(seed)
    seed = seed.reshape(1,seed.shape[0])


for i in generated_text:
    print(indices_word[i], end=" ")


