#!/usr/bin/python
# Guido Cnossen 

import sys
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from image_featurizer import face_detection, is_grey_scale

def featurize(image_path):
	
	image_vector = {}
	gray_scale, red, green, blue, average_RGB, brightness = is_grey_scale(image_path)
	geen_gezicht, een_gezicht, meerdere_gezichten, aantal_gezichten, glimlach, number_of_eyes, geen_bril, leesbril, gezichtsratio = face_detection(image_path)
	image_vector = {'Geen_gezicht:': geen_gezicht, 'Een_gezicht:': een_gezicht, 'Meerdere_gezichten:': meerdere_gezichten,'Aantal gezichten:': aantal_gezichten, 'Glimlach:': glimlach, 
	'Aantal_ogen': number_of_eyes, 'Gray_scale:':gray_scale, 'Red:': red, 'Green:': green, 'Blue:': blue, 'Average-RGB:': average_RGB, 'Brightness:' : brightness, 
	'Geen_bril:': geen_bril, 'Leesbril:': leesbril, 'Gezichtsratio:': gezichtsratio}
	
	return image_vector
	
	# I used the part below for testing my script
	'''
def main(argv):
	
	dictionary_list = []
	profile_pics = []
	text = open(argv[1],'r')
	for lines in text:
		line = lines.split()
		for x in line:
			x = x[27:]
			profile_pics.append(x)
			
	print(profile_pics)
	
	for image_path in profile_pics:
		print(image_path)
		dictionary_list.append(vectorize(image_path))
	
		
if __name__=='__main__':
	main(sys.argv)
	'''
