# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:09:10 2016

@author: mparmentier
"""

# Travail dans Mongo

from pymongo import MongoClient
from bson.son import SON
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

client = MongoClient('localhost', 27017)
db = client.MOPA
marches = db.Marches

def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text

print(marches.count())
print(marches.count({'ANNONCE.GESTION.INDEXATION.DESCRIPTEURS.DESCRIPTEUR.CODE': '116'}))

pipeline1 = [
{"$unwind": "$ANNONCE.GESTION.INDEXATION.DESCRIPTEURS.DESCRIPTEUR"},
{"$group": {"_id": "$ANNONCE.GESTION.INDEXATION.DESCRIPTEURS.DESCRIPTEUR.LIBELLE", "count": {"$sum": 1}}},
{"$sort": SON([("count", -1), ("_id", -1)])}
]

print("--- Aggrégation par descripteurs ---")
solution = marches.aggregate(pipeline1)
descr = list(solution)
print('Nombre de descripteurs différents : ' + str(len(descr)))
i = 1
chiffres = list()
labels   = list()
for value in descr:
    if(i<=20):
        print('Descripteur : ' + value['_id'] + '. Nombre : ' + str(value['count']))
        chiffres.append(float(value['count'] / len(descr)))
        labels.append(str(value['_id']))
    else:
        continue
    i = i+1
explode = (0,0,0,0,0,0,0,0,0,0,0,0,0,0.3,0,0,0,0,0,0)
plt.pie(chiffres, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')

fig = plt.figure()
plt.show()