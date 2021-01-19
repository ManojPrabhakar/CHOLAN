import os
import torch
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
import time
import sys
import string
import requests, json
import collections

def get_wikidata_id(wikipedia_title):
    wikidata_Qids_List = []

    for i, title in enumerate(wikipedia_title):
        try:
            url = 'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&titles=%s&format=json' % str(title)
            response = requests.get(url).json()['query']['pages']
            df = pd.io.json.json_normalize(response)
            df.columns = df.columns.map(lambda x: x.split(".")[-1])
            wikidata_id = df.get(key='wikibase_item').values

            wikidata_Qids_List.append(wikidata_id[0])
        except:
            print("Invalid Title - ", title)
            wikidata_Qids_List.append("NA")

    return wikidata_Qids_List


def readfile1(filename):
    '''
    read file
    '''
    f = open(filename)
    sentence_data = []
    entity_data = []
    wiki_title_data = []
    sentence = []
    entity = []
    wiki_title = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                entity_dict = collections.OrderedDict.fromkeys(entity)
                entity = list(entity_dict.keys())
                wiki_title_dict = collections.OrderedDict.fromkeys(wiki_title)
                wiki_title = list(wiki_title_dict.keys())
                #entity = list(set(entity))
                #wiki_title = list(set(wiki_title))

                sentence_data.append(sentence)
                entity_data.append(entity)
                wiki_title_data.append(wiki_title)
                sentence = []
                entity = []
                wiki_title = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0].strip('\n'))
        if len(splits) == 7 and (splits[1] == 'B'):
            entity.append(splits[2])
            wiki_title.append(splits[4][len("http://en.wikipedia.org/wiki/"):].replace('_', ' '))

    if len(sentence) > 0:
        #entity_dict = collections.OrderedDict.fromkeys(entity)
        #entity = list(entity_dict.keys())
        #wiki_title_dict = collections.OrderedDict.fromkeys(wiki_title)
        #wiki_title = list(wiki_title_dict.keys())
        sentence_data.append(sentence)
        entity_data.append(entity)
        wiki_title_data.append(wiki_title)
        sentence = []
        entity = []
        wiki_title = []
    return create_df(sentence_data, entity_data, wiki_title_data)


def create_df(sentence_list, entity_list, wiki_title_list):
    df_file = pd.DataFrame()
    total_entity_count = 0
    total_wiki_entity_count = 0
    for i in range(0, len(sentence_list)):
        #for j in range(0, len(sentence_list)-1):
        sentence = ' '.join(token for token in sentence_list[i])
        wikidata_id = get_wikidata_id(wiki_title_list[i])
        total_entity_count += len(wiki_title_list[i])
        if len(wikidata_id) == len(wiki_title_list[i]):
            total_wiki_entity_count += len(wikidata_id)
            # Print the count of entity aligned with the qids
            print("Sentence - ", i + 1, "\tActual_Entities - ", len(wiki_title_list[i]), "\tAligned_Entities - ", len(wikidata_id), "\tTotal_Entities - ", total_entity_count, "\tTotal_Aligned_Entities - ", total_wiki_entity_count)
            entity = ' '.join(entity + ' EntityMentionSEP' for entity in entity_list[i])
            wiki_title = ' '.join(wiki_title + ' WikiLabelSEP' for wiki_title in wiki_title_list[i])
            uri = ' '.join(qid for qid in wikidata_id)
            d = {'Sentence': sentence, 'Entity': entity, 'WikiTitle': wiki_title, 'Uri': uri}
            df_file = df_file.append(d, ignore_index=True)

    df_file = df_file.fillna('NIL_ENT')
    return df_file

if __name__ == '__main__':
    data_dir = "/data/prabhakar/CG/CONLL-AIDA/AIDA_data/"
    in_file = data_dir + "testb.txt"
    out_file = data_dir + "cholan_aida_testb.txt"

    #dataset = "ace2004"
    #data_dir = "/data/prabhakar/CG/WNED/"+ dataset +"/"
    #in_file = data_dir + dataset + "_sample.conll"
    #out_file = data_dir + dataset + "_cholan_test_sample.txt"

    #df_infile = pd.read_csv(in_file, sep='\t', encoding='utf-8')
    #readfile(in_file, out_file)

    df_file = readfile1(in_file)
    df_file.to_csv(out_file , sep='\t', encoding='utf-8', index=False)

    #wikipedia_title = ["Uzbekistan national football team", "Germany"]
    #wikidata_id = get_wikidata_id(wikipedia_title)
    #print(wikipedia_title, " - ", wikidata_id)