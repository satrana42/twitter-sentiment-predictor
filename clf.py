from sklearn import *
from scipy.sparse import *
from scipy import *
import cPickle as pickle
import nltk
import re
import sys

(clf,d,shape) = pickle.load(open("model.p","rb"))
regex_strip = re.compile(r"^[a-zA-Z\'\#\!\?\.]+$")
regex_split = re.compile(r"[\s,\-\(\)\!\?\"\:\;\.]")
regex_dot = re.compile(r"^[\.]+$")
regex_exc = re.compile(r"^[\!\?\.]+$")
neg_set = set(["not","dont","cant","wont","shouldnt","couldnt","havent","hasnt","hadnt","nor","never","neither","no","none"])
sep_set = set(["but","yet",",",".","!","?","still",";"])

def tokenize(b):
	token = nltk.word_tokenize(b)
	token = [t.lower() for t in token]
	token = [t for t in token if regex_strip.match(t)]
	neg=0
	for i in range(len(token)):
		if regex_dot.match(token[i]): token[i] = "tok_dot"
		if regex_exc.match(token[i]): token[i] = "tok_exc"
		if token[i] in sep_set: neg=0
		if neg==1: token[i] = "neg_"+token[i]
		if token[i].endswith("n't") or token[i] in neg_set: neg=1-neg
	#print token
	return token

def predict_tweet(tweet):
	tokens = tokenize(tweet.strip('\\').decode('utf8','replace'))
	mat = dok_matrix((1,shape),dtype=float64)
	for i in range(len(tokens)):
		if d.has_key(tokens[i]):
			mat[0,d[tokens[i]][0]]=d[tokens[i]][1]
		if(i<len(tokens)-1 and d.has_key(tokens[i]+" "+tokens[i+1])):
		 	cbig = tokens[i]+" "+tokens[i+1]
		 	mat[0,d[cbig][0]] = d[cbig][1] 
	return clf.predict(mat[0])[0]

while True:
	try:
		tweet = sys.stdin.readline()
	except KeyboardInterrupt:
		break
	if not tweet:
		break
	print predict_tweet(tweet)