#!/usr/bin/python3
import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import os

BUCKET_URL = 'http://ratos.s3-sa-east-1.amazonaws.com/'
DATASET_FOLDER = 'dataset/'

def download_arquivo(file):
    tamanho = len(file)
    local_filename = "{}{}".format(DATASET_FOLDER, file)
    if file[tamanho-1] == '/':
        try:
            os.mkdir(local_filename)
        except OSError:
            print("Falhou ao criar diretorio %s" % local_filename)
        else:
            print("Criou diretorio %s com sucesso" % local_filename)
    else:
        url = "{}{}".format(BUCKET_URL, file)
        urllib.request.urlretrieve(url, local_filename)
        print("Baixou arquivo %s com sucesso" % local_filename)

if __name__ == '__main__':
    # cria diretorio base
    try:
        os.makedirs(DATASET_FOLDER)
    except:
        print("Diretorio base já existe")
    else:
        print("Criou diretório base")

    # faz o download do bucker
    url = BUCKET_URL
    url = urllib.request.urlopen(url)
    data = url.read()
    # print(data.decode())
    tree = ET.fromstring(data.decode())
    for child in tree:
        for grandson in child:
            tag = grandson.tag
            if tag != None:
               if tag.find("Key") > 0:
                   file  = grandson.text
                   if (file != None) and (file.find("index.html")<0):
                    download_arquivo(file)