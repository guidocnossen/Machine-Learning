#!/usr/bin/python
# Guido Cnossen 

import urllib
import sys
import urllib.request
import json
from collections import defaultdict

def main(argv):
	json_files = []
	text = open(argv[1],'r')
	for lines in text:
		line = lines.split()
		for x in line:
			x = x[36:]
			json_files.append(x)
	
	with open("profile_names_personality.txt", "w") as text_file:
		for map in json_files:
			f = open("users_id/" + map)
			json_file = f.read()
			data = json.loads(json_file)
			user = data['user']['screen_name']
			user_id = data['user']['id']
			text_file.write("{0}\t{1}\n".format(user,user_id))
			try:
				url= "https://twitter.com/" + user + "/profile_image?size=original"	
				urllib.request.urlretrieve(url, "profilepics/" + user + ".png")
		
			except urllib.error.HTTPError:
				continue
			
	with open("MBTI_labels_personality.txt", "w") as text_file:
		t = open('TwiSty-NL.json')
		twisty_json = t.read()
		data2 = json.loads(twisty_json)
		keys = data2.keys()
		for i in keys:
			label = data2[i]['mbti']
			user_id = data2[i]['user_id']
			text_file.write("{0}\t{1}\n".format(user_id,label))
				
	
		
		
	
	text.close()
	t.close()
	f.close()

		
if __name__ == "__main__":
	main(sys.argv)
