#!/usr/bin/env python
# coding: utf-8

# In[139]:


import re
import numpy as np
import sys
import nltk
nltk.download('punkt')

# In[140]:


import json
import keras
import keras.utils
from keras import utils as np_utils

path_mod = sys.argv[1]
# path_mod = "./model2"
p1 = path_mod + '/Translator_English_w2i'
p2 = path_mod + '/Translator_French_w2i'
p3 = path_mod + '/Translator_4.json'
p4 = path_mod + '/Translator_4_weights.hdf5'


f = open(p1)
word_to_index = json.load(f)
f1 = open(p2)
word_to_index2 = json.load(f1)
with open(p3,'r') as f:
    model_load = f.read()
f.close()

model = keras.models.model_from_json(model_load)
model.load_weights(p4)


type(word_to_index)


# In[144]:


f = input("input sentence: ")
# f = open("./intro-to-nlp-assign3/ted-talks-corpus/train.en")
# f1 = open("./intro-to-nlp-assign3/europarl-corpus/test.europarl")
# f2 = open("./intro-to-nlp-assign3/ted-talks-corpus/train.fr")

# reviews = []
# for line in f:
#     line = line.rstrip()
#     reviews.append(line)
# full_text = " ".join(reviews)
# reviews2 = []
# for line in f2:
#     line = line.rstrip()
#     reviews2.append(line)
# full_text2 = " ".join(reviews2)
reviews = []
reviews.append(f)
# reviews


# In[145]:





# In[146]:


def tokenization(s) :
    
    ## [\?\.\!.......] will check the occurence of these punctuations
    ## ?= will see what is next in the string (lookahead)
    ## + will check multiple occurence
    s = re.sub(r'[\?\.\!\&\>\<\[\(\]\)\-]+(?=[\?\.\!\&\>\<\[\(\]\)\-])', '', s)

    # <URL>
    s = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', s, flags=re.MULTILINE)

    # <HASHTAG>
    s = re.sub('#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', s, flags=re.MULTILINE)

    # <MENTION>
    s = re.sub('@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', s, flags=re.MULTILINE)

    # WORD tokenizer
    # wrd = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",s)
    wrd = re.findall('[a-zA-Z0-9<>]+', s)

    return wrd


# In[147]:


tokenized_data = []
tokenized_data2 = []
for l1 in reviews:
        tokenized_data.append(tokenization(l1))
# for l1 in reviews2:
#         tokenized_data2.append(tokenization(l1))


# In[148]:


# print(tokenized_data[:5])
import copy
org_data = copy.deepcopy(tokenized_data)
# org_data2 = copy.deepcopy(tokenized_data2)
# org_data[0]


# In[149]:


def dofunc1():
    mx_len_e = 0
    for i in tokenized_data:
        if(len(i) > mx_len_e):
            mx_len_e = len(i)
    mx_len_f = 0
    for i in tokenized_data2:
        if(len(i) > mx_len_e):
            mx_len_f = len(i)
    print(mx_len_e,mx_len_f)
    mxlen = max(mx_len_e,mx_len_f)
    mxlen


# In[150]:


def dofunc2():
   for i in range(len(tokenized_data)):
       ldiff = mx_len_e - len(tokenized_data[i])
       for j in range(ldiff):
           tokenized_data[i].append('PAD')
   for i in range(len(tokenized_data2)):
       ldiff = mx_len_f - len(tokenized_data2[i])
       for j in range(ldiff):
           tokenized_data2[i].append('pad')


# In[151]:


# len(tokenized_data[100])


# In[152]:


def dofunc3():
    tokens = []

    for i in tokenized_data:
        tokens.append("<S>")
        tokens += i
        tokens.append("<E>")

    print(tokens[:100])

    freq = {
        i: 0
        for i in tokens
    }

    for token in tokens:
        freq[token] += 1

    # tokens = [i for i in tokens if freq[i] >= 5]
    ###############################################
    tokens2 = []

    for i in tokenized_data2:
        tokens2.append("<S>")
        tokens2 += i
        tokens2.append("<E>")

    print(tokens2[:100])

    freq2 = {
        i: 0
        for i in tokens2
    }

    for token2 in tokens:
        freq[token2] += 1

# tokens2 = [i for i in tokens2 if freq2[i] >= 2]


# In[153]:


def dofunc4():
    unique_tokens = list(sorted(set(tokens)))
    vocab_size = len(unique_tokens)
    unique_tokens2 = list(sorted(set(tokens2)))
    vocab_size2 = len(unique_tokens2)
    print(vocab_size,vocab_size2)


# In[154]:


# unique_tokens.append('UNKNOWN')
# unique_tokens.append('PAD')
# unique_tokens2.append('pad')


# print(index_to_word[0])
def dofunc5():
    word_to_index = {word: index+1 for index, word in enumerate(unique_tokens)} 
    index_to_word = {index+1: word for index, word in enumerate(unique_tokens)}
    word_to_index['PAD'] = 0
    index_to_word[0] = 'PAD'
    ###########################
    word_to_index2 = {word: index+1 for index, word in enumerate(unique_tokens2)} 
    index_to_word2 = {index+1: word for index, word in enumerate(unique_tokens2)}
    word_to_index2['pad'] = 0
    index_to_word2[0] = 'pad'


# In[155]:


# type(word_to_index2)
# index_to_word[1000]
# index_to_word2[1000]
# len(index_to_word2)


# In[156]:



# len(word_to_index)
# len(index_to_word)


# In[157]:


# [chk = 1 if 1 = 1]
# print(chk)


# In[158]:


# import spacy
# vecfnd = spacy.load('en_core_web_sm')
# wrdtovec = np.random.rand(len(word_to_index),300)
# for key in word_to_index:
#     if not bool(vecfnd(key)):
#         wrdtovec[key] = vecfnd(key).vector
# print(wrdtovec.shape)


# In[159]:


# print(wrdtovec[500],index_to_word[500])


# In[160]:


def dofunc6():
    from sklearn.model_selection import train_test_split
    # train_data , val_data = train_test_split(tokenized_data,test_size=0.2,random_state = 42)
    varlst = []
    for i in range(len(tokenized_data)):
        v1 = []
        for j in range(len(tokenized_data[i])):
            if(j < 20):
                v1.append(tokenized_data[i][j])
            else:
                break
        varlst.append(v1)
    train_data = varlst
    varlst = []
    for i in range(len(tokenized_data2)):
        v1 = []
        for j in range(len(tokenized_data2[i])):
            if(j < 20):
                v1.append(tokenized_data2[i][j])
            else:
                break
        varlst.append(v1)
    train_data2 = varlst


# In[161]:


# len(train_data)
# print(train_data[0][0])
# print(len(train_data2))
# train_data2[:-100]
# len(val_data)


# In[162]:


import random
def getidseq(inp):
    cur = []
    for j in inp:
        lst = []
        for i in j:
            if i not in word_to_index:
                v = word_to_index[random.choice(list(word_to_index.keys()))]
                lst.append(v)
            elif i in word_to_index:
                v = word_to_index[i]
                lst.append(v)
        cur.append(lst)
    return cur

# val_seq = getidseq(val_data)
# train_seq[:-5]
# train_seq = getidseq(train_data)
# train_seq2 = getidseq(train_data2)
# print(train_seq[:10])
# print(train_seq2[:100])
# print(len(train_seq),len(train_seq2))


# In[163]:


# len(train_seq[0])


# ## model

# In[164]:


# n = 5


# In[165]:


# tx,ty = prep_data(train_seq,n)


# In[166]:


# |print(len(tx),len(ty))


# In[167]:


# n_mx_e = 20
# n_mx_f = 20


# In[168]:


# len(unique_tokens)


# In[169]:


def dofunc7():
    from keras.layers import Embedding,LSTM,Dense,RepeatVector, TimeDistributed
    from keras.models import Sequential
    model = Sequential()
    model.add(Embedding(input_dim = len(unique_tokens),output_dim = 15,input_length = n_mx_e,mask_zero = True))

    model.add(LSTM(128))
    model.add(RepeatVector(n_mx_f))

    model.add(LSTM(128,return_sequences=True))
    model.add(Dense(len(unique_tokens2),activation = 'softmax'))
    model.compile('adam','categorical_crossentropy')


# In[170]:


# vv = [0]*5
# vv[2] = 5
# vv


# In[171]:


# def preohe(y):
#     mlst = []
#     for i in range(len(y)):
#         l1 = []
#         for j in range(len(y[i])):
#             vlst = [0]*len(unique_tokens2)
#             e = y[i][j]
#             vlst[e] = 1
#             l1.append(vlst)
#         mlst.append(l1)
#     return mlst
    
# batch_size = 100
# for i in range(0,len(train_seq),batch_size):
#     print(i)
#     curX = train_seq[i:i+batch_size]
#     cury = train_seq2[i:i+batch_size]
#     curY = preohe(cury)
#     model.fit(curX,curY,epochs = 10)


# In[172]:


# import json
# import keras
# import keras.utils
# from keras import utils as np_utils

# f = open('./aad/Translator_English_w2i')
# word_to_index = json.load(f)
# f1 = open('./aad/Translator_French_w2i')
# word_to_index2 = json.load(f1)
# with open('./aad/Translator_4.json','r') as f:
#     model_load = f.read()
# f.close()

# model = keras.models.model_from_json(model_load)
# model.load_weights('./aad/Translator_4_weights.hdf5')


# type(word_to_index)


# In[173]:


index_to_word = {}
for i in word_to_index:
    index_to_word[word_to_index[i]] = i
index_to_word
index_to_word2 = {}
for i in word_to_index2:
    index_to_word2[word_to_index2[i]] = i


# In[174]:


# print(getidseq(org_data[0]))
# print(org_data[0])
# print(org_data[0][0])
# cur = []
# for j in org_data[0]:
#     print(j)
#     for i in j:
#         print(i)
# for j in inp:
# #     print(j)
#     lst = []
#     for i in j:
#         if i not in word_to_index:
#             v = word_to_index[random.choice(list(word_to_index.keys()))]
#             lst.append(v)
#         elif i in word_to_index:
#             v = word_to_index[i]
#             lst.append(v)
#     cur.append(lst)


# In[175]:


# print(org_data)


# In[179]:


def translate_func(inp):
#     print(len(inp))
    pred = model.predict(np.array([inp]))
    fr_lst = []
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            mxval = 0
            for z in range(len(pred[i][j])):
                if(pred[i][j][z] > pred[i][j][mxval]):
                    mxval = z
            if index_to_word2[mxval] == '<pad>':
                break
            fr_lst.append(mxval)
    trans_lst = []
    for i in range(len(fr_lst)):
        trans_lst.append(index_to_word2[fr_lst[i]])
    return trans_lst
        
org_data_id = getidseq(org_data)
v = []
for i in range(len(org_data)):
#     print(i)
#     print(org_data[i])
    apv = []
#     print(org_data_id[i])
#     tlen = len(org_data_id)
    for j in range(0,len(org_data_id[i]),24):
        if(j+23 < len(org_data_id[i])):
            apv.append(translate_func(org_data_id[i][j:j+24]))
        else:
            v3 = j+24-len(org_data_id[i])
            for z in range(v3):
                org_data_id[i].append(0)
            apv.append(translate_func(org_data_id[i][j:j+24]))
    var1 = []
    for i in range(len(apv)):
        var1 = var1 + apv[i] 
#     var1 = var1[:-1]
    v.append(var1)

solstr = ""
for i in v[0]:
    solstr = solstr + str(i) + " "
print(solstr)

# bscore = []
# print(len(v))
# for i in range(len(v)):
#     print(i)
#     BLEUscore = nltk.translate.bleu_score.sentence_bleu([org_data2[i]], v[i])
#     bscore.append(BLEUscore)
# print(bscore[:10])


# In[76]:


# print(org_data[:5])
# print(org_data2[:5])


# In[37]:


# # corpus_bleu = nltk.translate.bleu_score.corpus_bleu(org_data2,v)
# import os
# f = open('./output2train.txt', 'w')

# # f.write(corpus_bleu)
# # f.write(os.linesep)

# for i in range(len(v)):
#     f.write(str(v[i]))
#     f.write("  ")
#     f.write(str(bscore[i]))
#     f.write(os.linesep)
    
# avgofall = sum(pmain)/len(pmain)
# f.write(str(avgofall))
# f.write(os.linesep)
# for i in range(len(reviews)):
#     f.write(reviews[i])
#     f.write("  ")
#     f.write(str(pmain[i]))
#     f.write(os.linesep)


# In[44]:


# import os
# f = open('./output2train2.txt', 'w')

# # f.write(corpus_bleu)
# # f.write(os.linesep)
# j = 100
# for i in range(len(org_data2)):
#     if(j > len(org_data2)-1):
#         j = 0
#     f.write(str(org_data2[i]))
#     f.write("  ")
#     f.write(str(bscore[j]))
#     f.write(os.linesep)
#     j = j + 1


# In[77]:


# len(org_data)


# In[78]:


# def translate_func(inp):
# #     print(len(inp))
#     pred = model.predict(np.array([inp]))
#     fr_lst = []
#     for i in range(len(pred)):
#         for j in range(len(pred[i])):
#             mxval = 0
#             for z in range(len(pred[i][j])):
#                 if(pred[i][j][z] > pred[i][j][mxval]):
#                     mxval = z
#             if index_to_word2[mxval] == '<pad>':
#                 break
#             fr_lst.append(mxval)
#     trans_lst = []
#     for i in range(len(fr_lst)):
#         trans_lst.append(index_to_word2[fr_lst[i]])
#     return trans_lst
        
# org_data_id = getidseq(org_data)
# vtest = []
# for i in range(len(org_data)):
#     print(i)
# #     print(org_data[i])
#     apv = []
# #     print(org_data_id[i])
# #     tlen = len(org_data_id)
#     for j in range(0,len(org_data_id[i]),24):
#         if(j+23 < len(org_data_id[i])):
#             apv.append(translate_func(org_data_id[i][j:j+24]))
#         else:
#             v3 = j+24-len(org_data_id[i])
#             for z in range(v3):
#                 org_data_id[i].append(0)
#             apv.append(translate_func(org_data_id[i][j:j+24]))
#     var1 = []
#     for i in range(len(apv)):
#         var1 = var1 + apv[i] 
# #     var1 = var1[:-1]
#     vtest.append(var1)
# def translate_func(inp):
# #     print(len(inp))
#     pred = model.predict(np.array([inp]))
#     fr_lst = []
#     for i in range(len(pred)):
#         for j in range(len(pred[i])):
#             mxval = 0
#             for z in range(len(pred[i][j])):
#                 if(pred[i][j][z] > pred[i][j][mxval]):
#                     mxval = z
#             if index_to_word2[mxval] == '<pad>':
#                 break
#             fr_lst.append(mxval)
#     trans_lst = []
#     for i in range(len(fr_lst)):
#         trans_lst.append(index_to_word2[fr_lst[i]])
#     return trans_lst
        
# org_data_id = getidseq(org_data)
# v = []
# for i in range(len(org_data)):
#     print(i)
# #     print(org_data[i])
#     apv = []
# #     print(org_data_id[i])
# #     tlen = len(org_data_id)
#     for j in range(0,len(org_data_id[i]),24):
#         if(j+23 < len(org_data_id[i])):
#             apv.append(translate_func(org_data_id[i][j:j+24]))
#         else:
#             v3 = j+24-len(org_data_id[i])
#             for z in range(v3):
#                 org_data_id[i].append(0)
#             apv.append(translate_func(org_data_id[i][j:j+24]))
#     var1 = []
#     for i in range(len(apv)):
#         var1 = var1 + apv[i] 
# #     var1 = var1[:-1]
#     v.append(var1)

# bscore = []
# print(len(v))
# for i in range(len(v)):
#     print(i)
#     BLEUscore = nltk.translate.bleu_score.sentence_bleu([org_data2[i]], v[i])
#     bscore.append(BLEUscore)
# print(bscore[:10])
# bscore2 = []
# print(len(vtest))
# for i in range(len(vtest)):
#     print(i)
#     BLEUscore = nltk.translate.bleu_score.sentence_bleu([org_data2[i]], vtest[i])
#     bscore2.append(BLEUscore)
# print(bscore2[:10])


# In[100]:


# import os
# f = open('./output2test2.txt', 'w')

# # f.write(corpus_bleu)
# # f.write(os.linesep)
# j = 100
# for i in range(len(org_data2)):
#     if(j > len(org_data2)-1):
#         j = 0
#     f.write(str(org_data2[i]))
#     f.write("  ")
#     f.write(str(bscore2[j]))
#     f.write(os.linesep)
#     j = j + 1


# In[ ]:




