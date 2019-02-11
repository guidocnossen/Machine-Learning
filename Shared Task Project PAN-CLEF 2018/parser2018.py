from xml.etree import ElementTree
from xml import etree
import re
import os
import sys

def find_files(directory):

    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(".xml")]


def CDATA(text=None):
    element = ElementTree.Element('![CDATA[')
    element.text = text
    return element

ElementTree._original_serialize_xml = ElementTree._serialize_xml

def _serialize_xml(write, elem, qnames, namespaces,short_empty_elements, **kwargs):

    if elem.tag == '![CDATA[':
        write("\n<{}{}]]>\n".format(elem.tag, elem.text))
        if elem.tail:
            write(_escape_cdata(elem.tail))
    else:
        return ElementTree._original_serialize_xml(write, elem, qnames, namespaces,short_empty_elements, **kwargs)

ElementTree._serialize_xml = ElementTree._serialize['xml'] = _serialize_xml

for fl in find_files("pan18-author-profiling-training-2018-02-27/"+sys.argv[1]+"/"+"text"):
	#in order to test it you have to create testing.xml file in the folder with the script
	xmlParsedWithET = ElementTree.parse(fl)
	root = xmlParsedWithET.getroot()
	text = open(fl).readlines() 
	print(fl)
	splitted_filename = fl.split('/')
	file_name = splitted_filename[-1].split('.')
	user_id = file_name[0]
	e = ElementTree.Element("data")
	cdata = CDATA(text)
	root.append(cdata)

	all_text =(root.getchildren()[1].text)

	sorted_text = "".join(all_text)
	end = sorted_text.replace("<document><![CDATA[", '')
	without_end= end.replace("]]></document>",'')
	start = without_end.split("]]></document>")
	new_tweet = without_end[34:]
	all_tweets = new_tweet.replace('\t', '')
	tweet= all_tweets[:-23]
	tweets = [tweet.rstrip()]
	account = tweets

	with open("test_tweet/"+"en"+"/"+user_id+"."+sys.argv[1]+".out", "w") as outfile:  
		outfile.write(','.join(str(s) for s in account)) 
	outfile.close()
tweets=[]

