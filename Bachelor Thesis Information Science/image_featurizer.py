#!/usr/bin/python
# Guido Cnossen 

from __future__ import division
import sys
import numpy as np
import cv2
from PIL import Image


def face_detection(image_path):
	
	img = cv2.imread(image_path)
	# call the correct haarcascade packages
	face_cascade = cv2.CascadeClassifier('OpenCV-tmp/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
	smile_cascade = cv2.CascadeClassifier('OpenCV-tmp/opencv/data/haarcascades/haarcascade_smile.xml')
	eye_cascade = cv2.CascadeClassifier('OpenCV-tmp/opencv/data/haarcascades/haarcascade_eye.xml')
	glasses_cascade = cv2.CascadeClassifier('OpenCV-tmp/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) #,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
	
	#set the featurevalues and other important values to 0
	number_of_eyes = 0
	geen_gezicht = 0
	een_gezicht = 0
	meerdere_gezichten = 0
	aantal_gezichten = 0
	gezichtsratio = 0 
	gezichtsratios = 0 # in case of more than one face
	geen_bril = 0
	zonnebril = 0
	leesbril = 0
	glimlach = 0
	gezichtsgrootte = 0
	
	#determine the size of the image for the 'gezichtsratio' feature
	height, width = tuple(img.shape[1::-1])
	image_size = height * width
	#detect the faces, determine the size of the face, determine the face-ratio
	for (x,y,w,h) in faces:
		aantal_gezichten += 1
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		
		#detect the eyes
		eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=15)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			number_of_eyes += 1
			if number_of_eyes >= (2*aantal_gezichten):
				number_of_eyes = (2*aantal_gezichten)
			if number_of_eyes == aantal_gezichten:
				number_of_eyes = number_of_eyes * 2
		
		#detect the smiles		
		smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor= 1.7, minNeighbors=22, minSize=(25, 25)) #,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
		for (x,y,w,h) in smile:
			cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
			glimlach += 1
		if glimlach >= aantal_gezichten:
			glimlach = aantal_gezichten
		if glimlach >= number_of_eyes:
			number_of_eyes = (2*aantal_gezichten)
		
		#detect if the person is wearing glasses
		glasses = glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=19)
		for (gx,gy,gw,gh) in glasses:
			cv2.rectangle(roi_color,(gx,gy),(gx+gw,gy+gh),(0,0,255),2)
			leesbril += 1		
		if leesbril != (number_of_eyes / 2) and number_of_eyes == (2* aantal_gezichten) and leesbril == 0:
			leesbril = 0
			geen_bril += 1
		if geen_bril >= 1:
			geen_bril = 1
		
	#cv2.imshow('img',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	#geen gezicht, een gezicht, meerdere gezichten
	if aantal_gezichten == 0:
		geen_gezicht += 1
	elif aantal_gezichten == 1:
		een_gezicht +=1
	else:
		meerdere_gezichten += 1
	
	# gezichtsratio
	if len(faces) == 0:
		gezichtsratio = 0
	elif len(faces) == 1:
		for i in faces:
			gezichtsgrootte = i[2] * i[3]
			gezichtsratio = float("{0:.2f}".format(gezichtsgrootte / image_size))
	else:
		for i in faces:
			gezichtsgrootte = i[2] * i[3]
			gezichtsratio = float("{0:.2f}".format(gezichtsgrootte / image_size))
			gezichtsratios += gezichtsratio 
		gezichtsratio = gezichtsratios / aantal_gezichten
	
	# divide the gezichtsratio values into classes to make the differences smaller between pictures
	if gezichtsratio <= 0.10:
		gezichtsratio  = 1
	elif gezichtsratio  >= 0.05 and gezichtsratio  <= 0.10:
		gezichtsratio  = 2
	elif gezichtsratio  >= 0.10 and gezichtsratio  <= 0.15:
		gezichtsratio  = 3
	elif gezichtsratio  >= 0.15 and gezichtsratio  <= 0.20:
		gezichtsratio = 4
	elif gezichtsratio  >= 0.20 and gezichtsratio  <= 0.25:
		gezichtsratio  = 5
	elif gezichtsratio  >= 0.25 and gezichtsratio  <= 0.30:
		gezichtsratio  = 6
	elif gezichtsratio  >= 0.30 and gezichtsratio  <= 0.35:
		gezichtsratio  = 7
	elif gezichtsratio  >= 0.35 and gezichtsratio  <= 0.40:
		gezichtsratio  = 8
	elif gezichtsratio  >= 0.40 and gezichtsratio  <= 0.45:
		gezichtsratio  = 9
	else:
		gezichtsratio = 10
	
	return [geen_gezicht, een_gezicht, meerdere_gezichten, aantal_gezichten, glimlach, number_of_eyes, geen_bril, leesbril, gezichtsratio]

def is_grey_scale(image_path):
	# RGB values
	Red = 0
	Green = 0
	Blue = 0
	pixel = 0
	Average_RGB = 0
	Brightness = 0
	gray_scale = 1
	im = Image.open(image_path).convert('RGB')
	w,h = im.size			
	for i in range(w):
		for j in range(h):
			r,g,b = im.getpixel((i,j))
			if r != g != b: 
				gray_scale = 0
			Red = Red + r
			Green = Green + g
			Blue = Blue + b
			pixel += 1
	
				
	# Average RGB
	Average_RGB = int((Red + Green + Blue) / pixel)
	
	Red = int(Red/pixel)
	Green = int(Green/pixel)
	Blue = int(Blue/pixel)
	
	# Brightness
	Brightness = int((Red + Green + Blue) / 3)
	
	# divide the avarage RGB, brightness, red, green, and blue values into classes to make the differences smaller between pictures
	#red classes
	if Red <= 25:
		Red = 1
	elif Red >= 25 and Red <=50:
		Red = 2
	elif Red >= 50 and Red <= 75:
		Red = 3
	elif Red >= 75 and Red <= 100:
		Red = 4
	elif Red >= 100 and Red <= 125:
		Red = 5
	elif Red >= 125 and Red <= 150:
		Red = 6
	elif Red >= 150 and Red <= 175:
		Red = 7
	elif Red >= 175 and Red <= 200:
		Red = 8
	elif Red >= 200 and Red <= 225:
		Red = 9
	else:
		Red = 10
	#green classes
	if Green <= 25:
		Green = 1
	elif Green >= 25 and Green <=50:
		Green = 2
	elif Green >= 50 and Green <= 75:
		Green = 3
	elif Green >= 75 and Green <= 100:
		Green = 4
	elif Green >= 100 and Green <= 125:
		Green = 5
	elif Green >= 125 and Green <= 150:
		Green = 6
	elif Green >= 150 and Green <= 175:
		Green = 7
	elif Green >= 175 and Green <= 200:
		Green = 8
	elif Green >= 200 and Green <= 225:
		Green = 9
	else:
		Green = 10
	#blue classes	
	if Blue <= 25:
		Blue  = 1
	elif Blue  >= 25 and Blue  <=50:
		Blue  = 2
	elif Blue  >= 50 and Blue  <= 75:
		Blue  = 3
	elif Blue  >= 75 and Blue  <= 100:
		Blue = 4
	elif Blue  >= 100 and Blue  <= 125:
		Blue  = 5
	elif Blue  >= 125 and Blue  <= 150:
		Blue  = 6
	elif Blue  >= 150 and Blue  <= 175:
		Blue  = 7
	elif Blue  >= 175 and Blue  <= 200:
		Blue  = 8
	elif Blue  >= 200 and Blue  <= 225:
		Blue  = 9
	else:
		Blue  = 10
	
	# Brightness classes
	if Brightness <= 25:
		Brightness  = 1
	elif Brightness  >= 25 and Brightness  <=50:
		Brightness  = 2
	elif Brightness  >= 50 and Brightness  <= 75:
		Brightness  = 3
	elif Brightness  >= 75 and Brightness <= 100:
		Brightness = 4
	elif Brightness  >= 100 and Brightness  <= 125:
		Brightness  = 5
	elif Brightness  >= 125 and Brightness  <= 150:
		Brightness  = 6
	elif Brightness  >= 150 and Brightness  <= 175:
		Brightness = 7
	elif Brightness  >= 175 and Brightness  <= 200:
		Brightness  = 8
	elif Brightness  >= 200 and Brightness  <= 225:
		Brightness  = 9
	else:
		Brightness  = 10
	
	# Average RGB classes	
	if Average_RGB <= 75:
		Average_RGB  = 1
	elif Average_RGB  >= 75 and Average_RGB  <=150:
		Average_RGB  = 2
	elif Average_RGB  >= 150 and Average_RGB  <= 225:
		Average_RGB  = 3
	elif Average_RGB  >= 225 and Average_RGB  <= 300:
		Average_RGB = 4
	elif Average_RGB  >= 300 and Average_RGB  <= 375:
		Average_RGB  = 5
	elif Average_RGB  >= 375 and Average_RGB  <= 450:
		Average_RGB  = 6
	elif Average_RGB  >= 450 and Average_RGB  <= 525:
		Average_RGB  = 7
	elif Average_RGB  >= 525 and Average_RGB  <= 600:
		Average_RGB  = 8
	elif Average_RGB  >= 600 and Average_RGB  <= 675:
		Average_RGB  = 9
	else:
		Average_RGB = 10
	
	return [gray_scale, Red, Green, Blue, Average_RGB, Brightness]
	
	
	# I used the part below for testing my script
'''	
def main(argv):

	image_vector = {}
	
	image_path = 'profilepics/xevabakker.png' 

	gray_scale, red, green, blue, average_RGB, brightness = is_grey_scale(image_path)
	geen_gezicht, een_gezicht, meerdere_gezichten, aantal_gezichten, glimlach, number_of_eyes, geen_bril, leesbril, gezichtsratio = face_detection(image_path)

	
	image_vector = {'Geen_gezicht:': geen_gezicht, 'Een_gezicht:':een_gezicht, 'Meerdere_gezichten:': meerdere_gezichten,'Aantal gezichten:': aantal_gezichten, 'Glimlach:': glimlach, 
	'Aantal_ogen': number_of_eyes, 'Gray_scale:':gray_scale, 'Red:': red, 'Green:': green, 'Blue:': blue, 'Average-RGB:': average_RGB, 'Brightness:' : brightness, 
	'Geen_bril:': geen_bril, 'Leesbril:': leesbril, 'Gezichtsratio:': gezichtsratio}
	print(image_vector)
	
	
if __name__== "__main__":
	main(sys.argv)
	
'''
