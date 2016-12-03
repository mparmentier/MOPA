# -*- coding: utf-8 -*-

import pandas as pd
import re
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import sys
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
import os
import datetime as dt
import re


CODE_DESCRIPTEUR = 274 # pour le choix de la categorie des documents ('ANNONCE.GESTION.INDEXATION.DESCRIPTEURS.DESCRIPTEUR.CODE')
NB_CLUSTERS = 5
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

# query pour mongo
# Note : certes, 'ANNONCE.GESTION.INDEXATION.DESCRIPTEURS.DESCRIPTEUR' est parfois un objet 'dictionnaire', et parfois une liste de dictionnaires
# cependant, en envoyant 'ANNONCE.GESTION.INDEXATION.DESCRIPTEURS.DESCRIPTEUR.CODE' à Mongo, ce dernier arrive automatiquement à gérer les 2 cas!
# du coup, mongo renvoie tous les documents dont (le code_descripteur unique) OU (l'un des codes descipteurs) est égal à CODE_DESCRIPTEUR
# c'est magique!
query = {
    'ANNONCE.GESTION.REFERENCE.IDWEB': {'$exists': 'true'},
    'ANNONCE.GESTION.INDEXATION.DESCRIPTEURS.DESCRIPTEUR.CODE': str(CODE_DESCRIPTEUR),
}
# selectionne les champs a rappatrier de mongo, pour eviter les transfers inutiles
projection = {
    'ANNONCE.DONNEES.OBJET.OBJET_COMPLET':1,
    'ANNONCE.GESTION.REFERENCE.IDWEB':1,
    #'ANNONCE.GESTION.INDEXATION.DESCRIPTEURS.DESCRIPTEUR.CODE': 1,
}
# requetage de mongo
try:
    result = db.Marches.find(query, projection).limit(MONGO_LIMIT)
except Exception as e:
    raise Exception("There is an error when querying mongo : " + e.message)

# Cette partie va retourner une liste de dictionnaires
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

print("Import Mongo : {} texts".format(len(texts)))

##############################
## Clustering
##############################

# conversion de texts en DataFrame. Cette ligne va convertir 'texts' en un tableau (dataframe) avec 2 colones : id et text
texts_df = pd.DataFrame(texts)

stop_words = get_stop_words('fr')

# Un HashingVectorizer découpe un texte en une liste de mots, et renvoie une matrice où chaque ligne correspond à
# un document et chaque colonne à un mot
# doc : http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
hasher = HashingVectorizer(strip_accents='unicode',
                               stop_words=stop_words,
                               norm=None)

# Le HashingVectorizer ne permet pas la stemmisation des mots durant le processus de tokenisation.
# On va donc lui dire de le faire quand même.
# Pour cela, on récupère sa fonction de tokenisation, que l'on va améliorer, puis lui réinjecter:

original_tokenizer = hasher.build_tokenizer() # recuperation de la fonction de tokenisation
stemmer = SnowballStemmer("french",  ignore_stopwords = True)

def new_tokenizer(text):
    words = original_tokenizer(text)
    stemmed_words = [stemmer.stem(w) for w in words]
    return stemmed_words

hasher = HashingVectorizer(tokenizer= new_tokenizer, # création d'un nouveau hasher avec injection de notre tokenizer amélioré
                           strip_accents='unicode',
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
cluster_ids = ["cl_" + str(i) for i in km.labels_]

# ajout de la colonne cluster_id à texts_df
texts_df['cluster_id'] = cluster_ids



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

# On relance une tfidf, cette fois ci dans le but de caractériser les clusters.
vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer= new_tokenizer, strip_accents='unicode')
X = vectorizer.fit_transform(texts_df['text'])
# on converti X en DataFrame
X = pd.DataFrame(X.toarray())

# On efectue la somme de chaque poids tfidf de chaque mot, pour l'ensemble des textes du cluster
X = X.groupby(cluster_ids).apply(sum)

features = vectorizer.get_feature_names()

# Pour chaque cluster, on prend les 20 mots dont le poids tfidf aggrégé est le plus fort
carac = X.apply(lambda arr: pickTopN(arr, features, 20), 1)

# creation du fichier CSV final
result = pd.DataFrame([[c] for c in carac],columns = ['key_words'])
result['cluster_id'] = carac.index
result['human_attributed_label'] = None # Ajout d'une colonne vide



##############################
## Enregistrement des fichiers
##############################

now = re.sub("[ :]",'-',str(dt.datetime.now())[:19])
dir_name = os.path.join('data','{}-DESCRIPTEUR_{}'.format(now,CODE_DESCRIPTEUR))

os.makedirs(dir_name, exist_ok=True)

result.to_csv(os.path.join(dir_name,'a_completer.csv'),index=False)

# copie de texts_df avec selection de 2 colonnes seulement
corresp = pd.DataFrame(texts_df,columns=['cluster_id','id'])
corresp.to_csv(os.path.join(dir_name,'corresp.csv'),index=False)

for cluster_id in X.index:
    texts_of_this_cluster = texts_df[texts_df.cluster_id == cluster_id]
    cluster_path = os.path.join( dir_name, cluster_id)
    os.makedirs(cluster_path)
    for text in texts_of_this_cluster.values:
        with open(os.path.join(cluster_path,text[0]),'w') as fi:
            words = re.split("\s",text[1])
            res = []
            for w in words:
                try:
                    res += [re.sub('[^\w]', '_', w, flags=re.UNICODE)]
                except Exception as e:
                    print('\n\nWarning: error when processing word "{}". This word will be excluded.\nOriginal\
                        exception : {}\n'.format(w, e.message))
            fi.write(' '.join(words))

print(result)

