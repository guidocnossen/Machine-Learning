import numpy as np
np.random.seed(113) #set seed before any keras import
from keras.models import Sequential
from keras.utils import np_utils
import sys
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Activation, Embedding, SimpleRNN, LSTM, Flatten, Conv1D, Dropout, MaxPooling1D, LSTM
from keras.models import Sequential
import string
from keras.optimizers import Nadam
from nltk.corpus import stopwords
from keras.preprocessing import sequence


def find_files(directory):

    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(".out")]


def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]

    accounts = " ".join(tokens)
    return accounts


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
        all_tweet = " ".join(text[:-1])
        all_tweets= clean_doc(all_tweet)
        male_users.append(all_tweets)
    else:
        all_tweet = " ".join(text[:-1])
        all_tweets= clean_doc(all_tweet)
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
num_classes = len(np.unique(y_train)) # how many labels we have
y_train_one_hot = y_train
y_test_one_hot = y_test
y_dev_one_hot = y_dev




# NEURAL NETWORK WITH DEEPLEARNING

w2i = defaultdict(lambda: len(w2i))
PAD = w2i["<pad>"] # index 0 is padding
UNK = w2i["<unk>"] # index 1 is for UNK

# convert words to indices, taking care of UNKs
X_train_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_train]

w2i = defaultdict(lambda: UNK, w2i) # freeze - cute trick!
X_dev_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_dev]
X_test_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_test]

max_account_length=max([len(s.split(" ")) for s in X_train] 
                        + [len(s.split(" ")) for s in X_dev] 
                        + [len(s.split(" ")) for s in X_test] )

# pad X
X_train_pad = sequence.pad_sequences(encoded_docs, maxlen=max_account_length, padding='post')
#X_train_pad = sequence.pad_sequences(X_train_num, maxlen=max_account_length, value=PAD)
X_dev_pad = sequence.pad_sequences(X_dev_num, maxlen=max_account_length, value=PAD)
#X_test_pad = sequence.pad_sequences(X_test_num, maxlen=max_account_length,value=PAD)
X_test_pad = sequence.pad_sequences(encoded_doc, maxlen=max_account_length, padding='post')


model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_account_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train_pad, y_train_one_hot,validation_split=0.1 , epochs=20, verbose=1, batch_size=100)
loss, accuracy = model.evaluate(X_test_pad, y_test)
EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
print("Accuracy: ", accuracy *100)




