import numpy as np
import sys
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

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


# MACHINE LEARNING


vectorizer = CountVectorizer(binary=True,analyzer='word')

## transform data to sklearn representation 
X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)
X_test_vec = vectorizer.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train_vec, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

y_pred = classifier.predict(X_dev_vec)



scores = cross_val_score(classifier, X_train_vec, y_train, cv=10 )
print(scores)
print("Accuracy cross-validation: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



print(accuracy_score(y_dev, y_pred))
