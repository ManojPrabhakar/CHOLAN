## Candidate Generation ##
from elasticsearch import Elasticsearch
import requests, json
from natsort import natsorted
import re

def initialization():
    es = Elasticsearch(['http://localhost:9200/'])
    docType = "doc"
    es.count()
    return es

def entity_search(query, size):
    es = initialization()
    indexName = "wikidataentityindex"
    match = []
    not_match = []
    #     print("docType - ", docType)
    ###################################################
    body = {
        "query": {
            "match": {
                "label": query,
            }
        }
        , "size": size
    }

    elasticResults = es.search(index=indexName, body=body)
    print("%d Hits :" % elasticResults['hits']['total'])

    for result in elasticResults['hits']['hits']:
        match.append([result["_source"]["label"], result["_source"]["uri"].strip("<http://www.wikidata.org/entity/>")])

        #match.append([result["_source"]["label"], result["_source"]["uri"], result["_score"]])
        #if result["_source"]["label"].lower() == query.replace(" ", "_").lower():
            #             print("## Equal ## - ", result["_source"]["label"], " - ", query.replace(" ", "_"))
        #    match.append([result["_source"]["label"], result["_source"]["uri"], result["_score"] * 50, 40])
        #    #match.append([result["_source"]["label"], result["_source"]["uri"]])
        #else:
            #             print("## Not Equal ## - ", result["_source"]["label"])
        #    match.append([result["_source"]["label"], result["_source"]["uri"], result["_score"] * 40, 0])
        #    #match.append([result["_source"]["label"], result["_source"]["uri"]])

    return match

def fuzzy_search(query, size):
    es = initialization()
    indexName = "wikidataentityindex"
    results = []
    body = {
        "query": {
            "fuzzy": {"label": query}
        }
        , "size": size
    }

    elasticResults = es.search(index=indexName, body=body)

    for result in elasticResults['hits']['hits']:
        if result["_source"]["label"].lower() == query.replace(" ", "_").lower():
            results.append([result["_source"]["label"], result["_source"]["uri"]])
        else:
            results.append([result["_source"]["label"], result["_source"]["uri"]])
    return results

def falcon_search(query):
    my_json = {"text": query, "spans": []}
    wikidata_URI_List =[]
    try:
        wikidata_URI_List = requests.post("https://labs.tib.eu/falcon/falcon2/api?mode=short&k=20", json=my_json).json()
        wikidata_URI = [uri[0].strip("<http://www.wikidata.org/entity/>") for uri  in wikidata_URI_List["entities"]]
        #wikidata_URI = '|'.join(wikidata_URI_List)
        return wikidata_URI
    except:
        print("No response")
        return ''


def sort(sub_list):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used

    #sub_list_sorted = sorted(sub_list, key=lambda x: x[1].strip("Q"))
    #sub_list_sorted = sub_list.sort()
    sub_list_sorted = natsorted(sub_list, key=lambda x: x[1].strip("Q"))
    return sub_list_sorted

def srt(val):
    """split and sort"""
    old = val.split(", ")
    new = ["{}{:0>2.0f}".format(i[0],int(i[1:]))  for i in old]
    new.sort()
    out = ", ".join([i for i in new])
    return out

def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.
    Required arguments:
    l -- The iterable to be sorted.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def append_list(finalList, lastList):
    for i in lastList:
        finalList.append(i)
    return finalList

if __name__ == '__main__':

    matchFinalList = []
    query = "India"  ## "Kuleshov"
    size = 40
    ## Normal Match ##
    match = entity_search(query, size)
    matchFirstList = match[:5]
    matchLastList = match[10:]
    matchLastListSorted = sort(matchLastList)
    matchFinalList = matchFirstList
    matchFinalList = append_list(matchFinalList, matchLastListSorted[:5])

    print("\n### Normal Match - Possible Candidate Items for ", query, "###")
    for i, x in enumerate(matchFinalList):
        print(i + 1, " - ", x[0], " - ", x[1].strip("<http://www.wikidata.org/entity/>"))

    ## Fuzzy Search ##
    match = fuzzy_search(query, size)
    print("\n### Fuzzy Search - Possible Candidate Items for ", query, "###")
    for i, x in enumerate(match):
        print(i + 1, " - ", x[0], " - ", x[1].strip("<http://www.wikidata.org/entity/>"))

    result = falcon_search(query)
    print("\n### Falcon Search - Possible Candidate Items for ", query, "###")
    print(result)