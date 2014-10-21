# coding=utf-8
from twokenize import *
import re
import cPickle as pickle
from scipy.sparse import *
from scipy import *
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn import *
import numpy as np
import random
from sklearn.utils import shuffle

# f = open('small_training.csv', 'r')
f = open('training.csv', 'r')

y = []
tokens = []
postags = []
bigrams = []
freq0, freq1, freq0b, freq1b = {} , {} , {} , {}
f0sz,f1sz,f0szb,f1szb = 0,0,0,0
tot=0
dsz = 0
dic = {}
rmp = {}
regex_strip = re.compile(r"^[a-zA-Z\'\#\!\?\.]+$")
regex_split = re.compile(r"[\s,\-\(\)\!\?\"\:\;\.]")
regex_dot = re.compile(r"^[\.]+$")
regex_exc = re.compile(r"^[\!\?\.]+$")
neg_set = set(["not","dont","cant","wont","shouldnt","couldnt","havent","hasnt","hadnt","nor","never","neither","no","none"])
sep_set = set(["but","yet",",",".","!","?","still",";"])
stopset = set(stopwords.words('english'))
stopset = stopset-neg_set-sep_set
stemmer=PorterStemmer()

def tokenize(b):
	#token = regex_split.split(b)
	token = nltk.word_tokenize(b)
	token = [t.lower() for t in token]
	token = [t for t in token if regex_strip.match(t)]
	# token = [stemmer.stem(t) for t in token if regex_strip.match(t) and t not in stopset]
	neg=0
	for i in range(len(token)):
		if regex_dot.match(token[i]): token[i] = "tok_dot"
		if regex_exc.match(token[i]): token[i] = "tok_exc"
		if token[i] in sep_set: neg=0
		if neg==1: token[i] = "neg_"+token[i]
		if token[i].endswith("n't") or token[i] in neg_set: neg=1-neg
	#print token
	return token

for line in f:
	a,b = int(line[1]), line[5:-2].strip("\\").decode("utf8","replace")
	cur_tokens = tokenize(b)
	tokens += [list(set(cur_tokens))]
	cur_big = []
	y += [a]
	for i in range(len(cur_tokens)):
		if(i<len(cur_tokens)-1):
			cur_big += [cur_tokens[i]+" "+cur_tokens[i+1]]
			if(a==0): 
				if not freq0b.has_key(cur_big[-1]):
					freq0b[cur_big[-1]]=0
				freq0b[cur_big[-1]]+=1
				f0szb+=1
			else:
				if not freq1b.has_key(cur_big[-1]):
					freq1b[cur_big[-1]]=0
				freq1b[cur_big[-1]]+=1
				f1szb+=1		
		if(a==0):
			if not freq0.has_key(cur_tokens[i]):
				freq0[cur_tokens[i]]=0	
			freq0[cur_tokens[i]]+=1
			f0sz+=1
		else: 
			if not freq1.has_key(cur_tokens[i]):
				freq1[cur_tokens[i]]=0
			freq1[cur_tokens[i]]+=1
			f1sz+=1
		
	bigrams += [list(set(cur_big))]
	#postags += nltk.pos_tag(cur_tokens)	

print f0sz,f1sz,f0szb,f1szb,len(freq0),len(freq1),len(freq0b),len(freq1b)

f0sz+=1
f1sz+=1
f0szb+=1
f1szb+=1

sortedf0, sortedf1, sortedf0b, sortedf1b = [], [], [] , []
for (item,cnt_item) in freq0.items():
	if freq0.has_key(item): cnt0 = freq0[item]
	else: cnt0 = 1 
	if freq1.has_key(item): cnt1 = freq1[item]
	else: cnt1 = 1 
	sz0,sz1=f0sz,f1sz
	sm0 = 1.*cnt0*(sz0+sz1)/((cnt0+cnt1)*sz0)
 	sm1 = 1.*cnt1*(sz0+sz1)/((cnt0+cnt1)*sz1)
 	sortedf0.append((item , 0.7*log2(sm0) - 0.3*log2(sm1)))

for (item,cnt_item) in freq0b.items():
	if freq0b.has_key(item): cnt0 = freq0b[item]
	else: cnt0 = 1 
	if freq1b.has_key(item): cnt1 = freq1b[item]
	else: cnt1 = 1 
	sz0,sz1 = f0szb,f1szb
	sm0 = 1.*cnt0*(sz0+sz1)/((cnt0+cnt1)*sz0)
 	sm1 = 1.*cnt1*(sz0+sz1)/((cnt0+cnt1)*sz1)
 	sortedf0b.append((item , 0.7*log2(sm0) - 0.3*log2(sm1)))

for (item,cnt_item) in freq1.items():
	if freq0.has_key(item): cnt0 = freq0[item]
	else: cnt0 = 1 
	if freq1.has_key(item): cnt1 = freq1[item]
	else: cnt1 = 1 
	sz0,sz1=f0sz,f1sz
	sm0 = 1.*cnt0*(sz0+sz1)/((cnt0+cnt1)*sz0)
 	sm1 = 1.*cnt1*(sz0+sz1)/((cnt0+cnt1)*sz1)
 	sortedf1.append((item , 0.7*log2(sm1) - 0.3*log2(sm0)))

for (item,cnt_item) in freq1b.items():
	if freq0b.has_key(item): cnt0 = freq0b[item]
	else: cnt0 = 1 
	if freq1b.has_key(item): cnt1 = freq1b[item]
	else: cnt1 = 1 
	sz0,sz1=f0szb,f1szb
	sm0 = 1.*cnt0*(sz0+sz1)/((cnt0+cnt1)*sz0)
 	sm1 = 1.*cnt1*(sz0+sz1)/((cnt0+cnt1)*sz1)
 	sortedf1b.append((item , 0.7*log2(sm1) - 0.3*log2(sm0)))

sortedf0 = sorted(sortedf0,key=operator.itemgetter(1))
sortedf1 = sorted(sortedf1,key=operator.itemgetter(1))
sortedf0b = sorted(sortedf0b,key=operator.itemgetter(1))
sortedf1b = sorted(sortedf1b,key=operator.itemgetter(1))
# sortedf0 = [s for s in sortedf0 if s[1]>0]
# sortedf0b = [s for s in sortedf0b if s[1]>0]
# sortedf1 = [s for s in sortedf1 if s[1]>0]
# sortedf1b = [s for s in sortedf1b if s[1]>0]
#print len(sortedf0), len(sortedf0b), len(sortedf1), len(sortedf1b)
#print sortedf0[-10:], sortedf0b[-10:], sortedf1[-10:], sortedf1b[-10:]
sortedf0 = sortedf0[-75000:]+sortedf0b[-25000:]
sortedf1 = sortedf1[-75000:]+sortedf1b[-25000:]
#sortedf0 = sorted(sortedf0,key=operator.itemgetter(1))
#sortedf1 = sorted(sortedf1,key=operator.itemgetter(1))

freq0 = {}
for (a,b) in sortedf0:
	freq0[a]=b

freq1 = {}
for (a,b) in sortedf1:
	freq1[a]=b

print "done"

x = dok_matrix((len(tokens),len(freq0)+len(freq1)),dtype=float64)

rcnt=0
for tokens_tweet in tokens:
	for token_tweet in tokens_tweet:
		if not freq0.has_key(token_tweet) and not freq1.has_key(token_tweet): continue
		if not dic.has_key(token_tweet):
			if freq0.has_key(token_tweet) and freq1.has_key(token_tweet):
				if freq0[token_tweet] > freq1[token_tweet]: ss=-abs(freq0[token_tweet])
				elif freq0[token_tweet] < freq1[token_tweet]: ss=abs(freq1[token_tweet])
				else: ss=0
			elif freq0.has_key(token_tweet): ss=-abs(freq0[token_tweet])
			else: ss=abs(freq1[token_tweet]) 
			dic[token_tweet] = (dsz,ss)
			dsz+=1
		x[rcnt,dic[token_tweet][0]]=dic[token_tweet][1]
	rcnt+=1

print "done"

rcnt=0
for bigs_tweet in bigrams:
	for big_tweet in bigs_tweet:
		if not freq0.has_key(big_tweet) and not freq1.has_key(big_tweet): continue
		if not dic.has_key(big_tweet):
			if freq0.has_key(big_tweet) and freq1.has_key(big_tweet):
				if freq0[big_tweet] > freq1[big_tweet]: ss=-abs(freq0[big_tweet])
				elif freq0[big_tweet] < freq1[big_tweet]: ss=abs(freq1[big_tweet])
				else: ss=0
			elif freq0.has_key(big_tweet): ss=-abs(freq0[big_tweet])
			else: ss=abs(freq1[big_tweet]) 
			dic[big_tweet] = (dsz,ss)
			dsz+=1
		x[rcnt,dic[big_tweet][0]]=dic[big_tweet][1]
	rcnt+=1


print "done"

#save = (y,x,dic)
#pickle.dump(save,open("small_save.p","wb"),1)

# clf = svm.LinearSVC().fit(x[10000:-10000],y[10000:-10000])
# # pickle.dump((clf,dic,shape(x)[1]),open("small_model.p","wb"),1)
# print "done"
# score  = clf.score(x[10000:-10000],y[10000:-10000])
# print "train ",score
# score  = clf.score(x[:10000],y[:10000])
# print "neg ",score
# score  = clf.score(x[-10000:],y[-10000:])
# print "pos ",score

clf = svm.LinearSVC().fit(x,y)
pickle.dump((clf,dic,shape(x)[1]),open("model.p","wb"),1)
print "done"
clf = svm.LinearSVC().fit(x[80000:-80000],y[80000:-80000])
score  = clf.score(x[80000:-80000],y[80000:-80000])
print "train ",score
score  = clf.score(x[-80000:],y[-80000:])
print "pos ",score
score  = clf.score(x[:80000],y[:80000])
print "neg ",score