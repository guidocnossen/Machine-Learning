# Guido Cnossen 

import sys
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

from image_featurizer import face_detection, is_grey_scale
from vectorizer import featurize

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import random

def show_most_informative_features(vectorizer, classifier, n=10):
    feature_names = vectorizer.get_feature_names()
    for i in range(0,len(classifier.coef_)):
        coefs_with_fns = sorted(zip(classifier.coef_[i], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        print("i",i)
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

def load_MBTI_labels_and_profilepics():
	# laad de profielfotonaam met de bijbehorende labels
	profile_pics = open('profile_names_personality.txt').readlines()
	labels = open('MBTI_labels_personality.txt').readlines() # -- classify the model over the complete label
	#labels = open('MBTI_labels_personality_one.txt').readlines() # -- classify the model over the first dimension of the MBTI-type (EXTRAVERT vs. INTROVERT)
	#labels = open('MBTI_labels_personality_two.txt').readlines() # -- classify the model over the second dimension of the MBTI-type (INTUITION vs. SENSING)
	#labels = open('MBTI_labels_personality_three.txt').readlines() # -- classify the model over the second dimension of the MBTI-type (THINKING vs. FEELING)
	#labels = open('MBTI_labels_personality_four.txt').readlines()	# -- classify the model over the second dimension of the MBTI-type (JUDGING vs. PERCEIVING)
	
	assert(len(profile_pics) == len(labels))
	data = list(zip(profile_pics, labels))
	
	random.shuffle(data)
	
	return data
	
def main(argv):
	
	random.seed(113)
	print('Loading data....')
	data = load_MBTI_labels_and_profilepics()
	
	print('Splitting data....')
	split_point = int(0.70 * len(data))
	split_point2 = int(0.85 * len(data))
	
	print('Tagging data....')
	profile_pics = [profile_pic for profile_pic, label in data]
	#sentences_tagged = [tag(str(sentence)) for sentence, _ in data]
	labels = [label for profile_pic, label in data]
	
	# kijken of ik de data kan splitten door middel van bovenstaande split_point manier te gebruiken in train, test etc.
	X_train, X_test, X_dev = profile_pics[:split_point], profile_pics[split_point:split_point2], profile_pics[split_point2:]
	#X_train_pos, X_test_pos = sentences_tagged[:split_point], sentences_tagged[split_point:]
	y_train, y_test, y_dev = labels[:split_point], labels[split_point:split_point2], labels[split_point2:]
	
	assert(len(X_train)==len(y_train))
	assert(len(X_test)==len(y_test))
	
	print('Vectorize data....')
	# featurize de profielfotos volgens functies die ik heb geschreven
	X_train_dict = []
	X_test_dict = []
	X_dev_dict = []
	
	# first extract the features (as dictionaries) ---- input voor beide moet een lijst met dictionaries zijn zoals in mijn functies wordt gedaan.
	# deze lijsten zijn gebaseerd op de vectors van alle items in X_train en X_test.
	for i in X_train:
		user = i.strip('\n')
		image_path = 'profilepics/' + user
		X_train_dict.append(featurize(image_path))
	print('Train dataset has been vectorized....')
	
	for i in X_test:
		user = i.strip('\n')
		image_path = 'profilepics/' + user
		X_test_dict.append(featurize(image_path))
	print('Test dataset has been vectorized....')
	
	# this part can also be used for the classifier, instead of the X_test set
	'''for i in X_dev:
		user = i.strip('\n')
		image_path = 'profilepics/' + user
		X_dev_dict.append(featurize(image_path))
	print('Dev dataset has been vectorized....')'''
	
	vectorizer = DictVectorizer()
	
	# then convert them to the internal representation (maps each feature to an id)
	X_train = vectorizer.fit_transform(X_train_dict)
	X_test = vectorizer.transform(X_test_dict)
	
	classifier = LogisticRegression()
	
	print("Training model..")
	classifier.fit(X_train, y_train)
	##
	print("Predict..")
	y_predicted = classifier.predict(X_test)
	###
	print("Accuracy:", accuracy_score(y_test, y_predicted))
	
	print('Classification report:')
	print(classification_report(y_test, y_predicted))
	
	print('Confusion matrix:')
	print(confusion_matrix(y_test, y_predicted))

	print('Most informative features:')
	show_most_informative_features(vectorizer, classifier, n=10)
	
if __name__ == '__main__':
	main(sys.argv)
