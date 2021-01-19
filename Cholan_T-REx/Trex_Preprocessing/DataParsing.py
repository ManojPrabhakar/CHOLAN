import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import string
import re
import tensorflow as tf
# nltk.download('punkt')
import nltk
import os
import pdb
from IPython.display import Markdown, display
from unicodedata import normalize
from DataSequence import limitSentLength

pd.set_option('display.max_colwidth', -1)

### Initialization ###
print("--Data Parsing--")
trex_path = '/data/prabhakar/manoj/arjun/dataset/Trex_raw/'
# trex_path = '../dataset/sample/trex_sample/'
entityData_Sep_dir = "/data/prabhakar/manoj/arjun/dataset/Trex_tsv_1/"
doc_id = 0

df = pd.read_csv('/data/prabhakar/manoj/arjun/dataset/entityData_Sep/' + 'WikidataLabel_clean.csv', encoding='utf-8', header=None, names=['qValue', 'entity'], sep='\t')
qDict = dict(zip(df.qValue, df.entity))


def parseDoc(docid, sentences, entityJson, sentences_boundaries):
    #global df_file
    global doc_id
    entity_list_dict = {entity['boundaries'][0]: entity['surfaceform'] for entity in entityJson
                        if entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker'
                        }
    entity_list_dict_sep = {entity['boundaries'][0]: entity['surfaceform'] + ' EntityMentionSEP' for entity in entityJson
                            if entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker'
                            }
    entity_list_dict_uri = {entity['boundaries'][0]: entity['uri'] for entity in entityJson
                            if entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker'
                            }
    entity_list = [entity_list_dict[key].strip() for key in sorted(entity_list_dict.keys())]
    entities = ' '.join(entity for entity in entity_list)

    entity_list_sep = [entity_list_dict_sep[key].strip() for key in sorted(entity_list_dict_sep.keys())]
    entities_sep = ' '.join(entity for entity in entity_list_sep)

    entity_list_uri = [entity_list_dict_uri[key].replace('http://www.wikidata.org/entity/', '').strip() for key in
                       sorted(entity_list_dict_uri.keys())]
    entities_uri = ' '.join(entity for entity in entity_list_uri)

    df = pd.DataFrame()
    if len(entity_list) == len(entity_list_uri):
        entity_uriEntity = [(entity.strip(), entityUri.strip()) for entity, entityUri in
                            zip(entity_list, entity_list_uri)]
        d = {'docid': docid.replace('http://www.wikidata.org/entity/', '').strip(), 'sequence1': sentences,
             'sequence2': entities, 'sequence2Sep': entities_sep, 'uri': entities_uri}
        df = pd.DataFrame(data=d, index=[0])
        df = df.fillna('')
    else:
        #         pass
        doc_id = doc_id + 1
    return df, entity_uriEntity


def parserFile(filename):
    df = pd.read_json(filename, encoding='utf-8')
    df = df[[u'docid', u'text', u'entities', u'sentences_boundaries']]
    df_sfile = pd.DataFrame()
    #     df.info()
    #     df.head()
    file_entity_uriEntity = []
    for i in range(len(df.index)):
        df_d, entity_uriEntity = parseDoc(df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3])
        file_entity_uriEntity = file_entity_uriEntity + entity_uriEntity
        df_sfile = df_sfile.append(df_d, ignore_index=True)
    return df_sfile, file_entity_uriEntity


'''# df_trex = df_trex.rename(columns = {"sequence1":"sentence", "sequence2Sep":"entitySep", "sequence2":"entity", "uri":"qid"} ) 
df_trex.head(2)
df_trex_new = df_trex[['sentence', 'entitySep', 'qid']]
df_trex_new.to_csv(entityData_Sep_dir + 'trex_dataset' + '.tsv', sep='\t', encoding='utf-8', index=False)
'''

def convertUriQvaluetoEntity(uris):
    strList = uris.strip().split()
    # print(uris[1])
    strList = [str(qDict[qValue]) + ' WikiLabelSEP' if qValue in qDict else 'NIL WikiLabelSEP' for qValue in strList]
    # print(strList[1])
    uris = ' '.join(strList)
    # print(uris[1])
    return uris



def main():
    df_file = pd.DataFrame()
    urlEntityDict = []

    filesList = sorted(os.listdir(trex_path))
    pool = mp.Pool(28)

    files_20 = [trex_path + file for file in filesList]
    print('\n\nNumber of JSON files = ', len(files_20))
    print(files_20)

    for result in pool.map(parserFile, files_20):
        startTime = time.time()
        print(len(result))
        df_file = df_file.append(result[0], ignore_index=True)
        elapsedTime = time.time() - startTime
        # urlEntityDict = merge_two_dicts(urlEntityDict, result[1])
        urlEntityDict = urlEntityDict + result[1]
        print("TimeTakenByFileParsing", elapsedTime)
        del result

    pool.close()
    pool.join()

    df_file.info()
    df_file.head(2)
    urlEntityDict


    df_file = df_file.dropna()
    df_file = df_file.drop_duplicates(keep='first')
    df_file = df_file.reset_index(drop=True)
    df_file.to_csv(entityData_Sep_dir + 'merged_465_entity_uri_sep' + '.tsv', sep='\t', encoding='utf-8', index=False)
    print(df_file.info())

    df_entity_uri = pd.DataFrame.from_records(urlEntityDict, columns=['Surface-Form', 'QUri'])
    df_entity_uri = df_entity_uri.dropna()
    df_entity_uri = df_entity_uri.drop_duplicates(keep='first')
    df_entity_uri = df_entity_uri.reset_index(drop=True)
    df_entity_uri.to_csv(entityData_Sep_dir + 'surfaceForm_wikiUri_sep' + '.tsv', sep='\t', encoding='utf-8',
                         index=False)
    print(df_entity_uri.info())

    del urlEntityDict
    del df_file
    del df_entity_uri


    ### Replacing WikiUri with WikiDataEntity ###

    chunks = pd.read_csv(entityData_Sep_dir + 'merged_465_entity_uri_sep.tsv', encoding='utf-8', chunksize=100000, sep='\t')

    df = pd.DataFrame()
    ctr = 0
    for chunk in chunks:
        chunk = chunk.dropna()
        print("Chunk = ", len(chunk))
        chunk['uriSequence2'] = chunk['uri'].apply(convertUriQvaluetoEntity)
        df = df.append(chunk, ignore_index=True)
        print("WikiLabels = ",len(df['uriSequence2']))
        df.to_csv(entityData_Sep_dir + 'merged_465_entity_uri_entity_sep_' + str(ctr) + '_.tsv', encoding='utf-8', index=False, sep='\t')

    print(df.info())

    df.to_csv(entityData_Sep_dir + 'merged_465_entity_uri_entity_sep.tsv', encoding='utf-8', index=False, sep='\t')

    print(df.info())
    display(df.head(2))

    del df
    #del qDict
    del chunks

    limitSentLength()


if __name__ == '__main__':
    main()
