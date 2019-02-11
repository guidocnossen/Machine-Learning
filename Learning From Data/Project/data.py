import pandas as pd
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv('hyperp-training-grouped.csv.xz',
					compression='xz',
					sep='\t',
					encoding='utf-8',
					index_col=0).dropna()

print("reading data...")
data = shuffle(data)

split_point = int(0.75*len(data))
train = data[:split_point]
test = data[split_point:]

print("shuffling and dividing data...")
train_list = list(zip(train.id,train.hyperp,train.bias,train.text))
test_list = list(zip(test.id,test.hyperp,test.bias,test.text))
	
print("writing data to outputfile...")

with open('trainset.txt', 'w') as f:
	for i in train_list:
		f.write(str(i[0]) + ' ' + str(i[1]) + ' ' + i[2] + ' ')
		j = i[3].split()
		for k in j:
			f.write(k + ' ')
		f.write('\n')

with open('testset.txt', 'w') as f2:
	for i in test_list:
		f2.write(str(i[0]) + ' ' + str(i[1]) + ' ' + i[2] + ' ')
		j = i[3].split()
		for k in j:
			f2.write(k + ' ')
		f2.write('\n')
f.close()
f2.close()
