# -*- coding: utf-8 -*-

import pandas as pd
import re
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
import sys
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline



NB_CLUSTERS = 30
MONGO_LIMIT = 2000

##############################
## Database
##############################

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
    result = db.Marches.find(query, projection).limit(MONGO_LIMIT)
except Exception as e:
    raise Exception("There is an error when querying mongo : " + e.message)

# Cette ligne va retourner une liste de dictionnaires
#texts = [{'id':r['ANNONCE']['GESTION']['REFERENCE']['IDWEB'], 'text':r['ANNONCE']['DONNEES']['OBJET']['OBJET_COMPLET']} for r in result]
#print(len(texts))

texts = []
for r in result:
    if 'OBJET_COMPLET' in r['ANNONCE']['DONNEES']['OBJET']:
        t = r['ANNONCE']['DONNEES']['OBJET']['OBJET_COMPLET']
    # parfois, 'objet_complet' est une liste...
    elif type(r['ANNONCE']['DONNEES']['OBJET']) == list:
        objets_complets = [oc['OBJET_COMPLET'] for oc in r['ANNONCE']['DONNEES']['OBJET']]
        # on concatène les descriptions en une seule string
        t = ' '.join(objets_complets)
    i = r['ANNONCE']['GESTION']['REFERENCE']['IDWEB']
    obj = {'id':i,'text':t}
    texts += [obj]



##############################
## Clustering
##############################

# conversion de texts en DataFrame. Cette ligne va convertir 'texts' en un tableau (dataframe) avec 2 colones : id et text
texts_df = pd.DataFrame(texts)

stop_words = [] #todo: remplir cette liste

# Un HashingVectorizer découpe un texte en une liste de mots, et renvoie une matrice où chaque ligne correspond à
# un document et chaque colonne à un mot
# doc : http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
hasher = HashingVectorizer(strip_accents='unicode',
                               stop_words=stop_words,
                               norm=None)

# Un pipeline est juste une liste dans laquelle on place différents processeurs.
# quand on injectera de la data dans le pipeline, la data passera par tous les processeurs, dans l'ordre
# ici, les textes passeront dans le hashvectorizer, puis la matrice qui en ressortira sera passée dans le calcul de tfidf
vectorizer = make_pipeline(hasher, TfidfTransformer())

# on injecte la data dans le pipeline, il en ressort une matrice tfidf
X = vectorizer.fit_transform(texts_df['text'])

# on prépare l'algorithme de clustering
km = MiniBatchKMeans(n_clusters=NB_CLUSTERS, init='k-means++', max_iter=1000, n_init=1,
                    init_size=1000, batch_size=5000, verbose=True)

# on lance l'algo sur la data
km.fit(X)

# recuperation des labels
cluster_ids = km.labels_



##############################
## Aide a la caracterisation des clusters
##############################

# retourne le top N des "features" donc la valeur est donnée dans "values"
def pickTopN(values, features, N):
    ind = pickTopNIndexes(values, N)
    return [features[i] for i in ind]

# retourne les indices des N nombres les plus grands dans "arr"
def pickTopNIndexes(arr, N):  # optimisable
    return arr.argsort()[-N:][::-1]


# on crée une copie de texts_df
labelled_texts = pd.DataFrame(texts_df)
# on ajoute une colonne contenant l'identifiant du cluster
labelled_texts['cluster_id'] = ["cl_" + str(i) for i in cluster_ids]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(texts_df['text'])
X = pd.DataFrame(X.toarray())

X = X.groupby(cluster_ids).apply(sum)
features = vectorizer.get_feature_names()

o = X.apply(lambda arr: pickTopN(arr, features, 15), 1)

print(o)

