#!/usr/bin/python
# Guido Cnossen 
#s2610833 

import urllib

import urllib.request

def main():
	user = 'Meerdanvermoeid' 
	url= "https://twitter.com/" + user + "/profile_image?size=original"
	
	urllib.request.urlretrieve(url, "profilepics/" + user + ".png")

main()
