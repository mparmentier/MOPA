# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:26:06 2016

@author: mparmentier
"""
#some ipython magic to show the matplotlib plots inline
#%matplotlib inline 

import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import MiniBatchKMeans
import sys

# Connexion
try:
    client = MongoClient('localhost', 27017)
    db = client.MOPA
    marches = db.Marches
    print("Connexion réussie")
except :
    sys.stderr.write("Erreur de connexion à MongoDB")
    sys.exit(1)

#stopwords = nltk.corpus.stopwords.words('french')
stemmer = SnowballStemmer("french",  ignore_stopwords = True)

stop_words=set()
with open("stop_words.txt") as f:
  for line in f:
    stop_words.add(line.rstrip('\r\n'))

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            #if not (token in stop_words):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

idweb = []
synopses = []
for data in db.Marches.find({'ANNONCE.GESTION.REFERENCE.IDWEB': {'$exists': 'true'}}).limit(500):
    try:
        idweb.append((data['ANNONCE']['GESTION']['REFERENCE']['IDWEB']))
        synopses.append(data['ANNONCE']['DONNEES']['OBJET']['OBJET_COMPLET'])
    except:
        continue

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    if not(i in stop_words):
        allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words=stop_words,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

dist = 1 - cosine_similarity(tfidf_matrix)

# Clustering
num_clusters = 8 # Cela va être les catégories
#km = KMeans(n_clusters=num_clusters)
# Modification le 11/08/2016
km = MiniBatchKMeans(n_clusters = 8, init='k-means++', n_init=1,
                     init_size=1000, batch_size=5000, verbose=0, compute_labels = 'True')
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

films = { 'title': idweb, 'synopsis': synopses, 'cluster': clusters}
frame = pd.DataFrame(films, index = [clusters] , columns = ['title', 'cluster'])
print(frame['cluster'].value_counts())

# grouped = frame['rank'].groupby(frame['cluster']) #groupby cluster for aggregation purposes
# print(grouped.mean()) #average rank (1 to 100) per cluster

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace
    
# Rajout le 11/08/2016
for k, centroid in enumerate(km.cluster_centers_):
    print ("Cluster #%d:" % k)
    print (" ".join([terms[i]
                   for i in centroid.argsort()[:-10 - 1:-1]]))

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#7570b3', 6: '#e7298a', 7: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'prestations, présent, marché, dans, fourniture', 
                 1: 'présent, prestations, marché, fourniture, dans', 
                 2: 'fourniture, marché, prestations, présent, dans', 
                 3: 'présent, marché, dans, prestations, fourniture', 
                 4: 'marché, présent, fourniture, prestations, dans',
                 5: 'dans, marché, prestations, présent, fourniture', 
                 6: 'prestations, marché, présent, dans, fourniture', 
                 7: 'présent, fourniture, marché, dans, prestations'}

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=idweb)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(36, 18)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)