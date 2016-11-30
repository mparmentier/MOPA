# -*- coding: utf-8 -*-

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



NB_CLUSTERS = 30




# Connexion
try:
    client = MongoClient('localhost', 27017)
    db = client.MOPA
    marches = db.Marches
    print("Connexion réussie")
except :
    sys.stderr.write("Erreur de connexion à MongoDB")
    sys.exit(1)

texts = []

# query pour mongo (celle que tu as écrite)
query = {'ANNONCE.GESTION.REFERENCE.IDWEB': {'$exists': 'true'}}
# selectionne les champs a rappatrier de mongo, pour eviter les transfers inutiles
projection = {
    'ANNONCE.DONNEES.OBJET.OBJET_COMPLET':1,
    'ANNONCE.GESTION.REFERENCE.IDWEB':1,
}
# requetage de mongo
try:
    result = db.Marches.find(query, projection).limit(500)
except Exception as e:
    raise Exception("There is an error when querying mongo : " + e.message)

# ceci s'appelle en python une "comprehension".
# c'est tres pratique, on écrit en une ligne une boucle FOR pour créer une liste:
# https://docs.python.org/2/tutorial/datastructures.html#list-comprehensions
# Cette ligne va retourner une liste de dictionnaires
texts = [{'id':r['ANNONCE']['GESTION']['REFERENCE']['IDWEB'], 'text':r['ANNONCE.DONNEES']['OBJET']['OBJET_COMPLET']} for r in result]
print(texts)

# conversion de texts en DataFrame. Cette ligne va convertir 'texts' en un tableau (dataframe) avec 2 colones : id et text
texts_df = pd.DataFrame(texts)

stop_words = [] #todo: remplir cette liste

# Un HashingVectorizer découpe un texte en une liste de mots, et renvoie une matrice où chaque ligne correspond à
# un document et chaque colonne à un mot
# doc : http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
hasher = HashingVectorizer(strip_accents='unicode',
                               stop_words='french',
                               norm=None)

# Un pipeline est juste une liste dans laquelle on place différents processeurs.
# quand on injectera de la data dans le pipeline, la data passera par tous les processeurs, dans l'ordre
# ici, les textes passeront dans le hashvectorizer, puis la matrice qui en ressortira sera passée dans le calcul de tfidf
vectorizer = make_pipeline(hasher, TfidfTransformer())

# on injecte la data dans le pipeline, il en ressort une matrice tfidf
X = vectorizer.fit_transform(data)

# on prépare l'algorithme de clustering
km = MiniBatchKMeans(n_clusters=NB_CLUSTERS, init='k-means++', max_iter=1000, n_init=1,
                    init_size=1000, batch_size=5000, verbose=True)

# on lance l'algo sur la data
km.fit(X)

