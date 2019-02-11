#!/usr/bin/python
# Guido Cnossen 

import urllib
import sys
import urllib.request
import json
from collections import defaultdict

def main(argv):
	profiles = open('profile_names_personality.txt', 'r')
	labels = open('MBTI_labels_personality.txt', 'r')
	
	profile_list = []
	labels_list = []
	new_list1 = []
	
	for lines in profiles:
		line = lines.split()
		profile_list.append(line)
	
	for lines in labels:
		line = lines.split()
		labels_list.append(line)
	
	# check the user_ids, compare them and write outputfiles in which the profilenames and the labels are written to
	with open("MBTI_labels_personality.txt", "w") as text_file:
		with open("profile_names_personality.txt", "w") as text_file2:
			for i in profile_list:
				for j in labels_list:
					if i[1] == j[0]:
						text_file.write("{0}\n".format(j[1]))
						text_file2.write("{0}.png\n".format(i[0]))

	profiles.close()
	labels.close()
if __name__ == "__main__":
	main(sys.argv)

