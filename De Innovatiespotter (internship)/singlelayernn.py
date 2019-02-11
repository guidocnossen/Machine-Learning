import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adagrad

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

import sys

# Read in the trefwoorden data
def read_corpus(corpus_file):
	print('Reading in data from {0}...'.format(corpus_file))
	words = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split()
			labels.append(parts[-1])
			words.append(parts[0:-1])
			
	print('Done!')
	
	return words,labels

def label_converter(value):
	
	# convert value to textual label presentation
	value = round(value)
	label = ''
	if value == 0:
		label = 'Agrifood'
	if value == 1:
		label = 'Chemie'
	if value == 2:
		label = 'Creatieve_Industrie'
	if value == 3:
		label = 'Energie'
	if value == 4:
		label = 'HTSM'
	if value == 5:
		label = 'Health'
	if value == 6:
		label = 'ICT'
	if value == 7:
		label = 'Logistiek'
	if value == 8:
		label = 'Water'
		
	return label
	
# ---------------------SINGLE LAYER NEURAL NETWORK CODE --------------------------

def main(argv):
	
	# Read in the data and embeddings
	X, Y = read_corpus('data/Topsectoren/' + argv[1])
	
	# spit the data into train and test set (--80%/20%)
	split_point = int(0.80*len(X))
	Xtrain = X[:split_point]
	Ytrain = Y[:split_point]
	Xtest = X[split_point:]
	Ytest = Y[split_point:]
	
	# create tokenizer to preprocess our text descriptions
	vocab_size = 12000
	tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
	tokenize.fit_on_texts(Xtrain)
	
	# feature 1: wide Bag of Words vocab_size vector
	bow_train = tokenize.texts_to_matrix(Xtrain)
	bow_test = tokenize.texts_to_matrix(Xtest)
	
	#convert string labels to one-hot encodings
	encoder = LabelEncoder()
	encoder.fit(Ytrain)
	Ytrain = encoder.transform(Ytrain)
	Ytest = encoder.transform(Ytest)
	num_classes = np.max(Ytrain) + 1
	
	# convert labels to one hot vector of variety categories
	Ytrain = keras.utils.to_categorical(Ytrain, num_classes)  
	Ytest = keras.utils.to_categorical(Ytest, num_classes)
	
	# Define the properties of model - layer, activation, loss-function, opimizer
	model = Sequential()
	model.add(Dense(input_shape = (vocab_size,), units = num_classes))
	model.add(Activation('sigmoid'))
	sgd = Adagrad()
	loss_function = 'cosine_proximity'
	model.compile(loss = loss_function, optimizer = sgd, metrics=['accuracy'])
	
	# Train the model 
	model.fit(bow_train, Ytrain, epochs = 10, batch_size = 80)
	loss, acc = model.evaluate(bow_test, Ytest)
	
	# Get accuracy scores and model summary
	Yguess = model.predict(bow_test)
	print(model.summary())
	print('Accuracies:')
	print('-'*80)
	print('Classification accuracy on test - Single Layer Neural Network:', acc)
	
	# print text, predicted label and actual label
	'''for i in range(len(predictions)):
		val = predictions[i].argmax()
		val2 = Ytest[i].argmax()
		print(Xtest[i])
		print('Predicted:', label_converter(val), 'Actual:', label_converter(val2), '\n')'''
	
	
if __name__ == '__main__':
	main(sys.argv)
