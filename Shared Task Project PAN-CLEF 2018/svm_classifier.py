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
import string

# using a seed for replicability
random.seed(113)


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
            all_tweet = " ".join(text[:-1])
            all_tweets= clean_doc(all_tweet)
            male_users.append(all_tweets)
        else:
            all_tweet = " ".join(text[:-1])
            all_tweets= clean_doc(all_tweet)
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

param_grid = dict(reduce_dim__n_components=[2, 5, 10], clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)

print(grid_search)

#crossvalidation
#print("cross-validation..")
#scores = cross_val_score(pipeline, X_train, y_train, cv=5 )
#print(scores)
#print("Accuracy cross-validation: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
