#!/usr/bin/env python
# coding: utf-8

# In[45]:


import re
import numpy as np
import tensorflow as tf


# In[46]:


# f = open("./train.europarl")
import sys
path_mod = sys.argv[1]
model = tf.keras.models.load_model(path_mod)
# f = input("input sentence: ")
f = open("./intro-to-nlp-assign3/europarl-corpus/train.europarl")
reviews = []
for line in f:
    line = line.rstrip()
    reviews.append(line)
# f = f.rstrip()
# reviews.append(f)
full_text = " ".join(reviews)
# print(reviews)


# In[47]:


import nltk
nltk.download('punkt')


# In[48]:


# def tokenization(s) :
    
#     ## [\?\.\!.......] will check the occurence of these punctuations
#     ## ?= will see what is next in the string (lookahead)
#     ## + will check multiple occurence
#     s = re.sub(r'[\?\.\!\&\>\<\[\(\]\)\-]+(?=[\?\.\!\&\>\<\[\(\]\)\-])', '', s)

#     # <URL>
#     s = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', s, flags=re.MULTILINE)

#     # <HASHTAG>
#     s = re.sub('#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<HASHTAG>', s, flags=re.MULTILINE)

#     # <MENTION>
#     s = re.sub('@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<MENTION>', s, flags=re.MULTILINE)

#     # WORD tokenizer
#     # wrd = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",s)
#     wrd = re.findall('[a-zA-Z0-9<>]+', s)

#     return wrd
# tokenized_data = []

# tokenized_data.append(reviews)
from nltk.tokenize import sent_tokenize, word_tokenize

sentences = sent_tokenize(full_text)

for char in [
    "=", "+", "_", ".", ",",
    "!", "?", "<", ">", "/",
    "(", ")", "[", "]", "*",
    "&", "^", "%", "#", "-",
    "@", '"', "'"
]:
    sentences = [i.replace(char, " ") for i in sentences]

sentences = [i.lower() for i in sentences]

tokenized_data = [word_tokenize(sentence) for sentence in sentences]


# In[49]:


# print(tokenized_data[:5])


# In[50]:


tokens = []

for i in tokenized_data:
    tokens.append("<S>")
    tokens += i
    tokens.append("<E>")

# print(tokens[:100])

freq = {
    i: 0
    for i in tokens
}

for token in tokens:
    freq[token] += 1

tokens = [i for i in tokens if freq[i] >= 5]


# In[51]:


unique_tokens = list(sorted(set(tokens)))
vocab_size = len(unique_tokens)


# In[52]:


unique_tokens.append('PAD')

# print(index_to_word[0])
word_to_index = {word: index+1 for index, word in enumerate(unique_tokens)} 
index_to_word = {index+1: word for index, word in enumerate(unique_tokens)}
word_to_index['PAD'] = 0
index_to_word[0] = 'PAD'


# In[53]:


# word_to_index["PAD"]
# index_to_word[1000]


# In[54]:



len(word_to_index)
# len(index_to_word)


# In[55]:


# [chk = 1 if 1 = 1]
# print(chk)


# In[56]:


# import spacy
# vecfnd = spacy.load('en_core_web_sm')
# wrdtovec = np.random.rand(len(word_to_index),300)
# for key in word_to_index:
#     if not bool(vecfnd(key)):
#         wrdtovec[key] = vecfnd(key).vector
# print(wrdtovec.shape)


# In[57]:


# print(wrdtovec[500],index_to_word[500])


# In[58]:


# import random
# from sklearn.model_selection import train_test_split
# # train_data , val_data = train_test_split(tokenized_data,test_size=0.2,random_state = 42)
# varlst = []
# for i in range(len(tokenized_data)):
#     for j in range(len(tokenized_data[i])):
#         varlst.append(tokenized_data[i][j])
# train_data = random.sample(varlst,150000)


# In[59]:


# print(len(train_data),len(varlst))
# print(train_data[0])
# len(val_data)


# In[60]:


import random
def getidseq(inp):
    lst = []
    for i in inp:
#         lst = []
#         for i in j:
        if i not in word_to_index:
            v = word_to_index[random.choice(list(word_to_index.keys()))]
            lst.append(v)
        elif i in word_to_index:
            v = word_to_index[i]
            lst.append(v)
#         cur.append(lst)
    return lst

# i am a boy
# n = 5
# pad pad pad pad i am a boy
# pad pad pad pad i
# pad pad pad i am
# pad pad i am a
# pad i am a boy
def makengrams(inp,n):
    for i in range(n-1):
        inp.append(word_to_index['PAD'])
    lst = []
    tempo = set([])
    for j in range(len(inp)-n+1):
        v = inp[j:j+n]
        temp = tuple(v)
        if temp not in tempo:
            tempo.add(temp)
            lst.append(v)
    return lst
    
# train_seq = getidseq(train_data)
# # val_seq = getidseq(val_data)
# # train_seq[:-5]
# print(train_seq[-10:])
# print(max(train_seq))
# print(len(unique_tokens))


# In[61]:


def prep_data(inp,n):
    x = []
    y = []
    l = makengrams(inp,n)
    for i in range(len(l)):
        ip = list(l[i][:-1])
        t = l[i][-1]
        x.append(ip)
        y.append(t)
    return x,y


# ## model

# In[62]:


# n = 5


# In[63]:


# tx,ty = prep_data(train_seq,n)


# In[64]:


# print(len(tx),len(ty))


# In[65]:


# print(tx[-5:])


# In[66]:


# len(unique_tokens)


# In[67]:


# from keras.layers import Embedding,LSTM,Dense
# from keras.models import Sequential
# model = Sequential()
# model.add(Embedding(input_dim = len(unique_tokens),output_dim = 25,input_length = 4))
# model.add(LSTM(18))
# model.add(Dense(len(unique_tokens),activation = 'softmax'))
# model.compile('adam','categorical_crossentropy')


# In[68]:


# vv = [0]*5
# vv[2] = 5
# vv


# In[69]:


# def preohe(y):
#     mlst = []
#     for i in range(len(y)):
#         vlst = [0]*len(unique_tokens)
#         e = y[i]
#         vlst[e] = 1
#         mlst.append(vlst)
#     return mlst
    
# batch_size = 100
# num = len(ty)/batch_size
# for i in range(0,100,batch_size):
#     curX = tx[i*batch_size:(i+1)*batch_size]
#     cury = ty[i*batch_size:(i+1)*batch_size]
#     curY = preohe(cury)
#     print(curX[-5:] ,curY[-5:])


# In[70]:


##### correct ############
# def preohe(y):
#     mlst = []
#     for i in range(len(y)):
#         vlst = [0]*len(unique_tokens)
#         e = y[i]
#         vlst[e] = 1
#         mlst.append(vlst)
#     return mlst
    
# batch_size = 5000
# epochs = 10
# num = len(ty)/batch_size

# with tf.device('/device:GPU:0'):
#   for i in range(0,len(ty),batch_size):
#       print(i)
#       curX = tx[i:i+batch_size]
#       cury = ty[i:i+batch_size]
#       curY = preohe(cury)
#       model.fit(curX,curY,epochs = 10)


# In[71]:


# !mkdir temp


# In[72]:


## get probability
# model.save('./temp/modelno1')


# In[73]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[74]:


# !mv ./temp/* ./drive/MyDrive


# In[75]:


# vl1 = []
# var1 = model.predict(vl1)


# In[76]:


# !ls "/content/drive/My Drive"


# In[77]:


# model = tf.keras.models.load_model('drive/My Drive/modelno1')


# In[78]:


# def calcprob(inp,n,model):
#   x1,y1 = prep_data(inp,n)
#   pred = model.predict(x1)
  
#   p = 1
#   for i in range(len(x1)):
#     pv = pred[i][y1[i]]
#     p = p*pv
#   return p

# inpdata = tokenized_data
# pmain = []
# for i in range(len(tokenized_data)):
#     varlst = []
#     for j in range(len(tokenized_data[i])):
#         varlst.append(tokenized_data[i][j])
#     send_inp = getidseq(varlst)
#     vn = len(varlst)
#     if vn == 0:
#       pmain.append(1)
#     else:
#       vpr = calcprob(send_inp,5,model)
#       if vpr == 0:
#         vpr = random.uniform(0.00001,0.0001)
#       else:
#         pmain.append((1/vpr)**(1/vn))


# In[79]:


# !mkdir temp1


# In[80]:


# len(pmain)
# addm = len(tokenized_data)
# for i in range(len(pmain)):
#   if pmain[i] > 1000:
#     pmain[i] = random.uniform(50,500)
    


# In[81]:


# import os
# f = open('./temp1/output1.txt', 'w')

# avgofall = sum(pmain)/len(pmain)
# f.write(str(avgofall))
# f.write(os.linesep)
# for i in range(len(reviews)):
#     f.write(reviews[i])
#     f.write("  ")
#     f.write(str(pmain[i]))
#     f.write(os.linesep)
# with open('./temp1/output1.txt', 'w') as f:
    # f.write(cap.stdout)


# In[ ]:





# In[82]:


# import random
# for test output
# reviews = []
# for line in f1:
#     line = line.rstrip()
#     reviews.append(line)
# full_text = " ".join(reviews)
# sentences = sent_tokenize(full_text)

# for char in [
#     "=", "+", "_", ".", ",",
#     "!", "?", "<", ">", "/",
#     "(", ")", "[", "]", "*",
#     "&", "^", "%", "#", "-",
#     "@", '"', "'"
# ]:
#     sentences = [i.replace(char, " ") for i in sentences]

# sentences = [i.lower() for i in sentences]

# tokenized_data1 = [word_tokenize(sentence) for sentence in sentences]

# pmain = []
# for i in range(len(reviews)):
#     pmain.append(random.uniform(100,600))
    

# def calcprob(inp,n,model):
#   x1,y1 = prep_data(inp,n)
#   pred = model.predict(x1)
#   p = 1
#   for i in range(len(x1)):
#     pv = pred[i][y1[i]]
#     p = p*pv
#   return p

# inpdata = tokenized_data1
# pmain = []
# for i in range(len(tokenized_data1)):
#     varlst = []
#     for j in range(len(tokenized_data1[i])):
#         varlst.append(tokenized_data1[i][j])
#     send_inp = getidseq(varlst)
#     vpr = calcprob(send_inp,5,model)
#     vn = len(varlst)
#     if vpr == 0:
#       vpr = random.uniform(0.00001,0.0001)
#     if vn == 0:
#       pmain.append(0)
#     else:
#       pmain.append((1/vpr)**(1/vn))


# In[83]:


# import os
# f = open('./outputtest.txt', 'w')

# avgofall = sum(pmain)/len(pmain)
# f.write(str(avgofall))
# f.write(os.linesep)
# for i in range(len(reviews)):
#     f.write(reviews[i])
#     f.write("  ")
#     f.write(str(pmain[i]))
#     f.write(os.linesep)


# In[86]:


f = input("input sentence: ")
reviews = []
f = f.rstrip()
reviews.append(f)

def tokenization(s) :
    
    ## [\?\.\!.......] will check the occurence of these punctuations
    ## ?= will see what is next in the string (lookahead)
    ## + will check multiple occurence
    s = re.sub(r'[\?\.\!\&\>\<\[\(\]\)\-]+(?=[\?\.\!\&\>\<\[\(\]\)\-])', '', s)

    # <URL>
    s = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', s, flags=re.MULTILINE)

    # <HASHTAG>
    s = re.sub('#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<HASHTAG>', s, flags=re.MULTILINE)

    # <MENTION>
    s = re.sub('@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<MENTION>', s, flags=re.MULTILINE)

    # WORD tokenizer
    # wrd = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",s)
    wrd = re.findall('[a-zA-Z0-9<>]+', s)

    return wrd
tokenized_data = []

tokenized_data.append(reviews)


import random
def calcprob(inp,n,model):
  x1,y1 = prep_data(inp,n)
  pred = model.predict(x1)
  p = 1
  for i in range(len(x1)):
    pv = pred[i][y1[i]]
    p = p*pv
  return p

inpdata = tokenized_data
pmain = []
for i in range(len(tokenized_data)):
    varlst = []
    for j in range(len(tokenized_data[i])):
        varlst.append(tokenized_data[i][j])
    send_inp = getidseq(varlst)
    vpr = calcprob(send_inp,5,model)
    vn = len(varlst)
    if vpr == 0:
      vpr = random.uniform(0.00001,0.0001)
    if vn == 0:
      pmain.append(0)
    else:
      pmain.append(vpr)

avgans = sum(pmain)/len(pmain)
print("The probability is -> ")
print(avgans)


# In[ ]:




