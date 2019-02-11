#!/usr/bin/python
# Guido Cnossen 

from __future__ import division
import urllib
import sys
import json
from collections import defaultdict


def main(argv):
	E = []
	text = open(argv[1],'r')
	for lines in text:
		line = lines.split()
		for x in line:
			if x == 'P':
				E.append(x)
	
	print(len(E)) 
	i = len(E)
	print(i)
	
	maj = (i/966) * 100
	print(maj)		
		
if __name__ == "__main__":
	main(sys.argv)
