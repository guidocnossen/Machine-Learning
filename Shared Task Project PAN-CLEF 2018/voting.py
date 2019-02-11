import sys
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix 
import numpy as np
import random
from sklearn import preprocessing
import pandas as pd
import os
import nltk
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import numpy as np
np.random.seed(113) #set seed before any keras import
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
import sys
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
from keras.callbacks import EarlyStopping
from string import punctuation
from nltk.corpus import stopwords
from keras import optimizers
from keras.optimizers import Adadelta
import string
from keras.constraints import maxnorm
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Activation, Embedding, SimpleRNN, LSTM, Flatten, Conv1D, Dropout, MaxPooling1D, LSTM
from keras.models import Sequential
import string
from keras.optimizers import Adam
from nltk.corpus import stopwords
from keras.preprocessing import sequence
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# using a seed for replicability
random.seed(113)

def tokenize(text):
    tknzr = TweetTokenizer()
    tokenized = tknzr.tokenize(text)
    return tokenized

def find_files(directory):

    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(".out")]


def load_sentiment_sentences_and_labels():
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

    le = preprocessing.LabelEncoder()
 
    male_account_labeld = \
    [(profile, "male" )
        for profile in male_users]    
    female_account_labeld = \
    [(profile, "female")
        for profile in female_users] 

    #seperate the labels and profiles
    male_labels = [labels for accounts,labels in male_account_labeld]
    female_labels = [labels for accounts,labels in female_account_labeld]


    male_lines = [accounts for accounts,labels in male_account_labeld]
    female_lines = [accounts for accounts,labels in female_account_labeld]

    #concatenate all data
    sentences = np.concatenate([male_lines, female_lines ], axis=0)
    labels = np.concatenate([male_labels, female_labels ], axis=0)

    #transform labels to numeric data
    transform = le.fit(labels)
    labels = le.transform(labels)

    # make sure to have a label for every data instance
    assert(len(sentences)==len(labels))
    data = list(zip(sentences,labels))
    random.shuffle(data)
    print("split data..", file=sys.stderr)
    split_point = int(0.80*len(data))
    
    #split the data in training and test data
    sentences = [sentence for sentence, label in data]
    labels = [label for sentence, label in data]
    X_train, X_test = sentences[:split_point], sentences[split_point:]
    y_train, y_test = labels[:split_point], labels[split_point:]

    assert(len(X_train)==len(y_train))
    assert(len(X_test)==len(y_test))

    return X_train, y_train, X_test, y_test

# read input data
X_train, y_train, X_test, y_test = load_sentiment_sentences_and_labels()
print("vectorize data..", file=sys.stderr)


#create pipeline, combining features with FeatureUnion
pipeline = Pipeline([('features', FeatureUnion([
                          
                          ('wrd', TfidfVectorizer(binary=False, max_df=1.0, min_df=2, norm='l2', sublinear_tf=True, use_idf=True, lowercase=True)),
                          ('char',TfidfVectorizer(analyzer='char', ngram_range=(3,6), binary=False, max_df=1.0, min_df=2, norm='l2', sublinear_tf=True, use_idf=True, lowercase=True))
                    ])),
                    ('clf', svm.LinearSVC(C=1.0))])

print("train model..")
pipeline.fit(X_train, y_train)

print("predict..")
y_predicted = pipeline.predict(X_test)

###
print("Accuracy for",sys.argv[1],':', accuracy_score(y_test, y_predicted))

# param_grid = dict(reduce_dim__n_components=[2, 5, 10], clf__C=[0.1, 10, 100])
# grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)

# print(grid_search)

#crossvalidation
# print("cross-validation..")
# scores = cross_val_score(pipeline, X_train, y_train, cv=5 )
# print(scores)
# print("Accuracy cross-validation: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



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


# first map all words to indices, then create n-hot vector
def convert_to_n_hot(X, vocab_size):
    out = []
    for instance in X:
        n_hot = np.zeros(vocab_size)
        for w_idx in instance:
            n_hot[w_idx] = 1
        out.append(n_hot)
    return np.array(out)



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


# NEURAL NETWORK WITH Traditional sparse n-hot encoding
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
# convert words to indices, taking care of UNKs
X_train_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_train]
w2i = defaultdict(lambda: UNK, w2i) # freeze
X_dev_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_dev]
X_test_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_test]


vocab_size = len(w2i)
X_train_nhot = convert_to_n_hot(X_train_num, vocab_size)
X_dev_nhot = convert_to_n_hot(X_dev_num, vocab_size)
X_test_nhot = convert_to_n_hot(X_test_num, vocab_size)


np.random.seed(113) #set seed before any keras import
model = Sequential()
model.add(Dense(200, input_shape=(vocab_size,), kernel_initializer='he_uniform',  kernel_constraint=maxnorm(5)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation('sigmoid'))
optimizer =Adadelta(lr=0.22)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

model.fit(X_train_nhot, y_train,validation_split=0.1 , epochs=15, verbose=1, batch_size=128)
loss, accuracy = model.evaluate(X_test_nhot,y_test)
EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
print("Accuracy: ", accuracy *100)




feed_forward = model.predict_classes(X_test_nhot, verbose=1)



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
model.add(Dense(10,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())
model.fit(X_train_pad, y_train_one_hot,validation_split=0.1 , epochs=10, batch_size=60, verbose=1)
loss, accuracy = model.evaluate(X_test_pad, y_test)
EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
print("Accuracy: ", accuracy *100)

lstm = model.predict_classes(X_test_pad, verbose=1)


print("svm")
svm_array = np.array(y_predicted)
svm_array = svm_array.reshape(len(svm_array),1)
svm_array.shape
print("lstm")
lstm_array = np.array(lstm)
lstm_array.shape
print("ff")
feed_forward_array =np.array(feed_forward)

X_all_predictions = np.array([svm_array, feed_forward, lstm])


# samples, nx, ny = all_predictions.shape

# X_all_predictions = all_predictions.reshape((samples, nx*ny ))

print("Naive bayes")
NBclf = MultinomialNB()
print("Train Naive bayes")
NBclf.fit(X_all_predictions, y_train)
y_predicted_NB = NBclf.predict(X_test)
print("Accuracy for",sys.argv[1],':', accuracy_score(y_test, y_predicted_NB))


print("train SVM model..")
clf_svm= svm.LinearSVC()
clf_svm.fit(X_all_predictions, y_train)

print("predict..")
y_predicted_SVM=  clf_svm.predict(X_test)
###
print("Accuracy for",sys.argv[1],':', accuracy_score(y_test, y_predicted_SVM))

print("train LR model..")
classifier_LR = LogisticRegression()
classifier_LR.fit(X_all_predictions, y_train)
print("predict..")
y_predicted_LR = classifier.predict(X_test)
print("Accuracy for",sys.argv[1],':', accuracy_score(y_test, y_predicted_LR))
