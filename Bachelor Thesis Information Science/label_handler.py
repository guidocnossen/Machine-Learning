#!/usr/bin/python
# Guido Cnossen 

import urllib
import sys
import json
from collections import defaultdict

def main(argv):
	labels = open('MBTI_labels_personality.txt', 'r')
	
	with open("MBTI_labels_personality_one.txt", "w") as text_file:
		
		with open("MBTI_labels_personality_two.txt", "w") as text_file1:
			with open("MBTI_labels_personality_three.txt", "w") as text_file2:
				with open("MBTI_labels_personality_four.txt", "w") as text_file3:
					for lines in labels:
						line = lines[0]
						line1 = lines[1]
						line2 = lines[2]
						line3 = lines[3]
						text_file.write("{0}\n".format(line))
						text_file1.write("{0}\n".format(line1))
						text_file2.write("{0}\n".format(line2))
						text_file3.write("{0}\n".format(line3))	
	
	# check the user_ids, compare them and write outputfiles in which the profilenames and the labels are written to
	#with open("MBTI_labels_personality.txt", "w") as text_file:
		

	labels.close()
if __name__ == "__main__":
	main(sys.argv)
