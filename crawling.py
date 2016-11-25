# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:40:42 2016

@author: mparmentier
"""

from pymongo import MongoClient
import os
import zipfile
import tarfile
import shutil
import xmltodict
import codecs

"""
Connexion au serveur ftp
"""
from ftplib import FTP
host     = "echanges.dila.gouv.fr"
user     = "anonymous"
password = "test"
port     = 21

ftp = FTP(host,user, password)
ftp.login()
ftp.cwd('/BOAMP/2016')

"""
Connexion à la base Mongo
"""
client = MongoClient('localhost', 27017)
db = client.MOPA
    
def xml_files(members):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".xml":
            yield tarinfo

def dezip(filezip, pathdst = ''): 
    if pathdst == '': pathdst = os.getcwd()  ## on dezippe dans le repertoire locale 
    zfile = zipfile.ZipFile(filezip, 'r') 
    for i in zfile.namelist():  ## On parcourt l'ensemble des fichiers de l'archive 
        print (i)
        if os.path.isdir(i):   ## S'il s'agit d'un repertoire, on se contente de creer le dossier 
            try: os.makedirs(pathdst + os.sep + i)
            except: pass
        else: 
            try: os.makedirs(pathdst + os.sep + os.path.dirname(i)) 
            except: pass 
            data = zfile.read(i)                   ## lecture du fichier compresse 
            fp = open(pathdst + os.sep + i, "wb")  ## creation en local du nouveau fichier 
            fp.write(data)                         ## ajout des donnees du fichier compresse dans le fichier local
            fp.close()
            
            archive = tarfile.open(i,'r:tar')
            archive.debug = 1    # Affiche les fichiers en cours de décompression
            files = xml_files(archive)
            archive.extractall(path='fichiers', members=files)
            archive.close()
            
    zfile.close()

def download(j):
    print ("Fichier =>",files[j])
    
    fichier = {"name": files[j]}
    enregistrements.insert_one(fichier)

    fhandle = open(files[j], 'wb')
    ftp.retrbinary('RETR ' + files[j], fhandle.write)
    fhandle.close()
    
def supprimer(j):
    os.remove(j)
    shutil.rmtree('fichiers')
    
def saving():
    marches = db.Marches
    path_xml = "fichiers"
    nb_fichiers_non_integres = 0
    for path, dirs, file_xml in os.walk(path_xml):
        resultats = []
        for element_xml in file_xml:
            xml = os.path.basename(path + '/' + element_xml)
            doc_file = codecs.open(os.path.dirname(path + '/' + element_xml + '/' + xml),"r", 'utf-8')
            try:
                doc = doc_file.read()
            except:
                nb_fichiers_non_integres = 1 + nb_fichiers_non_integres
                continue
            dictionnaire = xmltodict.parse(doc)
            resultats.append(dictionnaire)
            doc_file.close()
        try:
            marches.insert_many(resultats)
            print(marches.count())
        except:
            continue
    print("Nombre de fichiers non intégrés : %s" % nb_fichiers_non_integres)
    
files = ftp.nlst()
    
enregistrements = db.enregistrements

for j in range(len(files)):
    if enregistrements.find_one({ "name": files[j] }) is None :
        download(j)
        for element in os.listdir():  
            if element.endswith('.DS_Store') or element.endswith('.xml') or element.endswith('.idea') or element.endswith('.py') or element.endswith('.tar'): 
                continue
                print("'%s' n'est pas un fichier texte. Je ne travaille pas dessus." % element)
            else:
                dezip(element)
                saving()
                supprimer(element)
    else:
        print(files[j])
        print("Déjà enregistré")