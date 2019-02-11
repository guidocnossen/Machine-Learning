from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import sys
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
np.random.seed(113) #set seed before any keras import
from keras.layers import  Flatten
from keras import regularizers

print('Loading data...')
def find_files(directory):

    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(".out")]


"""
loads the data
"""
en_user_list=[]
ar_user_list=[]
pt_user_list=[]
es_user_list=[]
with open("data/pan18-author-profiling-training-2018-02-27/en/en.txt", "r") as outfile:
    for line in outfile:
        user = line.rstrip().split(":::")
        en_user_list.append(user)
outfile.close()

with open("data/pan18-author-profiling-training-2018-02-27/ar/ar.txt", "r") as outfile:
    for line in outfile:
        user = line.rstrip().split(":::")
        ar_user_list.append(user)
outfile.close()
with open("data/pan18-author-profiling-training-2018-02-27/es/es.txt", "r") as outfile:
    for line in outfile:
        user = line.rstrip().split(":::")
        es_user_list.append(user)
outfile.close()

female_users = []
male_users = []
for fl in find_files("data/tweets/" + sys.argv[1]):
    text = open(fl).readlines()
    filename = fl[15:]
    file_name = filename.split('.')

    if sys.argv[1] == "ar":
        userlist = ar_user_list
    elif sys.argv[1] == "en":
        userlist = en_user_list
    elif sys.argv[1] == "es":
        userlist = es_user_list

    for i in userlist:
        if i[0] == file_name[0]:
            text.append(i)
    if text[-1][1] == "male":
        all_tweets = " ".join(text[:-1])
        male_users.append(all_tweets)
    else:
        all_tweets = " ".join(text[:-1])
        female_users.append(all_tweets)

male_account_labeld = \
[(profile, 0)
    for profile in male_users]    
female_account_labeld = \
[(profile, 1)
    for profile in female_users] 

#seperate the labels and profiles
male_labels = [labels for accounts,labels in male_account_labeld]
female_labels = [labels for accounts,labels in female_account_labeld]


male_lines = [accounts for accounts,labels in male_account_labeld]
female_lines = [accounts for accounts,labels in female_account_labeld]

#concatenate all data

sentences = np.concatenate([male_lines,female_lines], axis=0)
labels = np.concatenate([male_labels,female_labels],axis=0)

## make sure we have a label for every data instance
assert(len(sentences)==len(labels))
data={}
np.random.seed(113) #seed
data['target']= np.random.permutation(labels)
np.random.seed(113) # use same seed!
data['data'] = np.random.permutation(sentences)


X_rest, X_test, y_rest, y_test = train_test_split(data['data'], data['target'], test_size=0.2)
X_train, X_dev, y_train, y_dev = train_test_split(X_rest, y_rest, test_size=0.2)
del X_rest, y_rest
print("#train instances: {} #dev: {} #test: {}".format(len(X_train),len(X_dev),len(X_test)))

t = Tokenizer()
t.fit_on_texts(X_train)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(X_train)
encoded_doc = t.texts_to_sequences(X_test)

w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
X_train_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_train]
w2i = defaultdict(lambda: UNK, w2i) # freeze
X_dev_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_dev]
X_test_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_test]

max_account_length=max([len(s.split(" ")) for s in X_train] 
                        + [len(s.split(" ")) for s in X_dev] 
                        + [len(s.split(" ")) for s in X_test] )

traning_len= 500

# pad X
X_train_pad = sequence.pad_sequences(encoded_docs, maxlen=traning_len, padding='post')
#X_dev_pad = sequence.pad_sequences(X_dev_num, maxlen=traning_len, padding='post')
X_test_pad = sequence.pad_sequences(encoded_doc, maxlen=traning_len, padding='post')



# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.twitter.27B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 200))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('Build model...')

# create the model
embedding_vecor_length = 200
model = Sequential()
model.add(Embedding(vocab_size, embedding_vecor_length, weights=[embedding_matrix], input_length=traning_len))
model.add(LSTM(100))
#model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train_pad, y_train, epochs=10, validation_split=0.1 ,batch_size=128)
# Final evaluation of the model

loss, accuracy = model.evaluate(X_test_pad, y_test)
EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
print("Accuracy: ", accuracy *100)
