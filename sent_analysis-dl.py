#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 10:11:53 2019

@author: priyal
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, Activation, GlobalMaxPooling1D, GRU
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

def create_network_cnn():
    model_glove = Sequential()
    e = Embedding(num_words, 200, weights=[embedding_matrix], input_length=40, trainable=True)
    model_glove.add(e)

    model_glove.add(Dropout(0.2))
    model_glove.add(Conv1D(filters=200, kernel_size=2, padding='valid', activation='relu', strides=1))
    model_glove.add(MaxPooling1D(pool_size=2))
    model_glove.add(Flatten())
    model_glove.add(Dropout(0.5))
    model_glove.add(Dense(1,activation='sigmoid'))
    model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model_glove.fit(padded_docs, np.array(y_train), batch_size=128, epochs=10, verbose=2)
    return model_glove

def create_network_lstm():
    model = Sequential()
    e = Embedding(num_words, 200, weights=[embedding_matrix], input_length=40, trainable=True)
    model.add(e)
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_network_cnnlstm():
    model_hy = Sequential()
    model_hy.add(Embedding(vocabulary_size, 200, input_length=40, weights=[embedding_matrix], trainable=True))
    model_hy.add(LSTM(4, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))
    model_hy.add(Conv1D(16,4,activation='relu'))
    model_hy.add(Flatten())                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    model_hy.add(Dense(units=1, activation='sigmoid'))
    model_hy.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_hy

f = open("result.txt", "r")
corpus = []
data=[]
for line in f:
    cols = line.split("\t")
    corpus.append(cols[2])
    if(cols[0]>cols[1]):#positive tweets
        cols[0]=1
    elif (cols[1]>cols[0]):#negative tweets
        cols[0]=0
    else:
        pass
        
    data.append(cols[0])
f.close()
corpus=data
# x_train, x_val, y_train, y_val = train_test_split(corpus, data_labels, test_size=0.20, train_size=0.80, random_state=1234)

# ------------------------------------GLOVE classifier---------------------------------------

# embeddings
embeddings_index = dict()
f = open("glove.twitter.27B.200d.txt", encoding="utf8")
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
t = Tokenizer()
t.fit_on_texts(corpus)
encoded_docs = t.texts_to_sequences(corpus) 
vocabulary_size = len(t.word_index) + 1

for x in corpus[:5]:
    print(x)
for x in encoded_docs[:5]:
    print(x)


padded_docs = pad_sequences(encoded_docs, maxlen=40)
print(padded_docs[:5])


num_words = vocabulary_size
embedding_matrix = np.zeros((num_words, 200))
for word, i in t.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


neural_network_cnn = KerasClassifier(build_fn=create_network_cnn, epochs=10,  batch_size=128, verbose=0)

print(cross_val_score(neural_network_cnn, padded_docs, data, scoring="accuracy", cv=10))
