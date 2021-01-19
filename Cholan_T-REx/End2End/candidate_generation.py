## Candidate Generation ##
from elasticsearch import Elasticsearch
import requests, json

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
        match.append([result["_source"]["label"], result["_source"]["uri"]])

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

def fuzzy_search(query):
    es = initialization()
    indexName = "wikidataentityindex"
    results = []
    body = {
        "query": {
            "fuzzy": {"label": query}
        }
        , "size": 10
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
        wikidata_URI_List = requests.post("https://labs.tib.eu/falcon/falcon2/api?mode=short&k=10", json=my_json).json()
        wikidata_URI = [uri[0].strip("<http://www.wikidata.org/entity/>") for uri  in wikidata_URI_List["entities"]]
        #wikidata_URI = '|'.join(wikidata_URI_List)
        return wikidata_URI
    except:
        print("No response")
        return ''


if __name__ == '__main__':

    query = "GPLed"  ## "Kuleshov"

    ## Normal Match ##
    match = entity_search(query, 10)
    print("\n### Normal Match - Possible Candidate Items for ", query, "###")
    for i, x in enumerate(match):
        print(i + 1, " - ", x[0], " - ", x[1].strip("<http://www.wikidata.org/entity/>"))

    ## Fuzzy Search ##
    match = fuzzy_search(query)
    print("\n### Fuzzy Search - Possible Candidate Items for ", query, "###")
    for i, x in enumerate(match):
        print(i + 1, " - ", x[0], " - ", x[1].strip("<http://www.wikidata.org/entity/>"))

    result = falcon_search(query)
    print("\n### Falcon Search - Possible Candidate Items for ", query, "###")
    print(result)