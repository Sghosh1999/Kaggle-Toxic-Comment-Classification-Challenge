# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:10:55 2020

@author: Sayantan
"""

"""
Dataset link : kaggle jigsaw toxic comments Classification Challenge
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model,Sequential
from keras.layers import Dense,Embedding, Input , Activation
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, optimizers, layers
from sklearn.metrics import  roc_auc_score


#Loading the Train_test data
train = pd.read_csv('jigsaw-toxic-comment-classification-challenge/train/train.csv')
test = pd.read_csv('jigsaw-toxic-comment-classification-challenge/test/test.csv')

#Listing down the Classes
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

#Train and test labels
list_sequences_train = train["comment_text"]
list_sequences_test = test["comment_text"]

max_features = 22000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sequences_train))

#Tokenizing and Indexing the comments
list_tokenized_train = tokenizer.texts_to_sequences(list_sequences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sequences_test)

#Defining the train and test sequences
#200 is the maximum length of the inut stream. Samll sentence =0, long sentences will be trimmed
maxlen = 200
X_train = pad_sequences(list_tokenized_train, maxlen = maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen = maxlen)

totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]

#Visualizing the Distribution of the words
plt.hist(totalNumWords,bins = np.arange(0,410,10))
plt.show()



#Step1: Adding the first Input Layer (None,200)
inp = Input(shape=(maxlen, ))

#Step2: Adding the embedding Layer (None,200,128)
#128 is a tunable params(Creating Word Embeddings)
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

#Step3: Defining the LSTM Layer with 60 Output
x = LSTM(60, return_sequences=True, name='lstm_layer')(x)

#Step4: Global Max Pooling Layer to convert 3D tensor into 2D
x = GlobalMaxPooling1D()(x)

#Step5: Adding a Dropout Reguralisation
x = Dropout(0.1)(x)

#Step6: Adding a Dense Layer
x = Dense(50, activation = 'relu')(x)

x = Dropout(0.1)(x)

#Step7: Final Output Layer
x = Dense(6, activation="sigmoid")(x)


model=Model(inputs=inp, outputs = x)

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

batch_size = 32
epochs = 3

model.fit(X_train, y, batch_size = batch_size, epochs = epochs, validation_split=0.1)

model.summary()

#getting prediction sfor the Submission File
y_test = model.predict(X_test)

sample_submission = pd.read_csv('jigsaw-toxic-comment-classification-challenge/sample_submission/sample_submission.csv')

sample_submission[list_classes] = y_test
sample_submission.to_csv('submission1.csv', index = False)












