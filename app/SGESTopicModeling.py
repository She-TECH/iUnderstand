#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import json
import re
import gzip
import spacy
import os

import gensim
from gensim import corpora

import nltk
from nltk import FreqDist

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)
    
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


# In[6]:


dataset = pd.read_csv('D:/SGES.csv')
dataset.head()


# In[7]:


# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()


# In[15]:


freq_words(dataset['comments'])


# In[8]:


# remove unwanted characters, numbers and symbols
dataset['comments'] = dataset['comments'].str.replace("[^a-zA-Z#]", " ")


# In[9]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[10]:


# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

# remove short words (length < 3)
dataset['comments'] = dataset['comments'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# remove stopwords from the text
comments = [remove_stopwords(r.split()) for r in dataset['comments']]

# make entire text lowercase
comments = [r.lower() for r in comments]


# In[22]:


freq_words(comments, 20)


# In[11]:


nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output


# In[15]:


tokenized_comments = pd.Series(comments).apply(lambda x: x.split())
print(tokenized_comments[1])

for c in range(len(tokenized_comments)):
    print(tokenized_comments[c])


# In[16]:


comments_2 = lemmatization(tokenized_comments)
print(comments_2[1]) # print lemmatized comments

for c in range(len(comments_2)):
    print(comments_2[c])


# In[17]:


comments_3 = []
for i in range(len(comments_2)):
    comments_3.append(' '.join(comments_2[i]))

dataset['comments'] = comments_3

freq_words(dataset['comments'], 10)


# In[19]:


dictionary = corpora.Dictionary(comments_2)


# In[21]:


doc_term_matrix = [dictionary.doc2bow(com) for com in comments_2]

# Preview BOW for our sample preprocessed document
document_num = 20
bow_doc_x = doc_term_matrix[document_num]

for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                     dictionary[bow_doc_x[i][0]], 
                                                     bow_doc_x[i][1]))


# In[22]:


# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=10, random_state=100,
                chunksize=1000, passes=50)


# In[44]:


lda_model.print_topics()


# In[45]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis


# In[ ]:




