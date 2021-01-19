import pandas as pd
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

from candidate_generation import *


def read_ent_desc():
    ent_desc_file = "/data/prabhakar/CG/CONLL-AIDA/ent2desc.json"
    print("---Reading Entity Description File ---", ent_desc_file)
    with open(ent_desc_file, "r") as read_file:
        ent_desc = json.load(read_file)
        entities = list(ent_desc.keys())
    return ent_desc, entities


def get_ent_desc(wiki_label, cand_ent_desc):
    try:
        cand_ent = wiki_label.strip().replace(" ", "_")
        EntityDescription = " ".join(cand_ent_desc[cand_ent])
    except:
        return "NA"
    return EntityDescription


def load_DCA_candidates(dataset):
    local_cand_file = "/data/prabhakar/CG/DCA/" + dataset + ".tsv"
    print("--- Loading Candidate File ---", local_cand_file)
    df_candidates = pd.read_csv(local_cand_file, sep='\t', encoding='utf-8', usecols=['mention', 'gold', 'candidates',
                                                                                      'context', 'mtype'])
    return df_candidates


def search_DCA_candidates(mention, df_candidates):
    candidates = []
    mention = mention.rstrip(' ')
    df_match = df_candidates['candidates'][(df_candidates.mention == mention)].head(1)
    try:
        if df_match.empty != True:
            candidates = [row.split(' | ') for row in df_match]
        if len(candidates[0]) >= 5:
            return candidates[0][0:5]
        return candidates[0]
    except:
        return "NA"


def process_dataset(df, size, local_candidates_flag, dataset):
    # Load the dataset into a pandas dataframe.
    #df = pd.read_csv("/data/prabhakar/CG/Trex_data/Trex_train_50000.tsv",  sep='\t', encoding='utf-8', usecols=['sequence1','sequence2Sep','uri','uriSequence2'])
    df = df.dropna()
    df = df.rename(columns={"Sentence": "Sentence", "predictedEnt": "EntitySep", "Uri": "Qid", "WikiTitle": "WikiLabel", "Entity":"TargetEnt"})
    df['label'] = int(1)

    df_target = pd.DataFrame()
    df_final = pd.DataFrame()

    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    print('df',df.columns)

    #wiki_label_search = ''
    Target_EntityMentions = [row['TargetEnt'].split('EntityMentionSEP') for index,row in df.iterrows()]
    Target_Qids = [row['Qid'].split() for index,row in df.iterrows()]
    Target_WikiLabels = [row['WikiLabel'].split('WikiLabelSEP') for index, row in df.iterrows()]
    Predicted_EntityMentions = [row['EntitySep'].split('EntityMentionSEP') for index, row in df.iterrows()]

    #for index, row in df.iterrows():
    #    print("Sentence - ", index)
    #    Predicted_EntityMentions = row['EntitySep'].split('EntityMentionSEP')

    cand_ent_desc, cand_entities = read_ent_desc()
    df_candidates = load_DCA_candidates(dataset)

    ent = 0

    for i in range(0, df.shape[0]):
        for j in range(0,len(Target_WikiLabels[i])-1):
            ent += 1
            print('Sentence - ', i+1, ', Entity - ', j+1, ', Total_Entity - ', ent)
            POS_FLAG = False
            #CG_query = Target_EntityMentions[i][j]
            if Target_WikiLabels[i][j] != '':
                CG_query = Target_EntityMentions[i][j]
                CG_result = []
                CG_result_set = ()

                if local_candidates_flag == True:
                    CG_result = search_DCA_candidates(CG_query, df_candidates)
                    CG_result_set = set(CG_result)
                else:
                    CG_result = entity_search(CG_query, size)
                    CG_result_set = set(tuple(x) for x in CG_result)

                df_intermediate = pd.DataFrame(columns=['sequence1', 'sequence2', 'label'])
                df_target_intermediate = pd.DataFrame()

                seq1 = Target_EntityMentions[i][j] + ' | ' + df.iloc[i]['Sentence']
                #seq1 = Target_EntityMentions[i][j]
                #target_seq2 = Target_WikiLabels[i][j] + ' | ' + Target_Qids[i][j]
                EntityDescription = get_ent_desc(Target_WikiLabels[i][j], cand_ent_desc)
                #target_seq2 = Target_WikiLabels[i][j] + ' | ' + EntityDescription
                target_seq2 = Target_WikiLabels[i][j]
                #df_intermediate = df_intermediate.append({'sequence1': seq1, 'sequence2': target_seq2, 'label': int(0)}, ignore_index=True)
                #target_sequence = seq1 + ' | ' + target_seq2
                if j < len(Target_EntityMentions[i])-1:
                    df_tgt = df_target_intermediate.append({'EntityMention': Target_EntityMentions[i][j] , 'Sentence': df.iloc[i]['Sentence'], 'Target_Qid': Target_Qids[i][j], 'Target_Wikilabel': Target_WikiLabels[i][j], 'label': int(1)}, ignore_index=True)
                    df_target = df_target.append(df_tgt)

                for r, result in enumerate(CG_result_set):
                    if local_candidates_flag == True:
                        result_wikilabel = result
                    else:
                        result_wikilabel = result[0]
                        result_qid = result[1].strip("<http://www.wikidata.org/entity/>")
                    #pred_seq2 = result_wikitarget_seq2label + ' | ' + result_qid
                    EntityDescription = get_ent_desc(result_wikilabel, cand_ent_desc)
                    #pred_seq2 = result_wikilabel.replace('_', ' ')  + ' | ' + EntityDescription
                    pred_seq2 = result_wikilabel.replace('_', ' ')

                    if result_wikilabel != Target_WikiLabels[i][j]:
                        if POS_FLAG == False:
                            df_GT = df_intermediate.append({'sequence1': seq1, 'sequence2': target_seq2, 'label': int(1)}, ignore_index=True)
                            #df_final = df_final.append(df_GT)
                            POS_FLAG = True
                        df_CG = df_intermediate.append({'sequence1': seq1, 'sequence2': pred_seq2, 'label': int(0)}, ignore_index=True)
                        df_final = df_final.append(df_CG)

                #if (result_qid != Target_Qids[i][j] and POS_FLAG == False):
                #    sequence = seq1 + ' | ' + pos_seq2
                #    df_pos = df_intermediate.append({'sequence' : sequence, 'label' : int(1)}, ignore_index=True)
                #    df_final = df_final.append(df_pos)
                #    POS_FLAG=True
                #elif(result_qid != Target_Qids[i][j]):
                #    sequence = seq1 + ' | ' + neg_seq2
                #    df_neg = df_intermediate.append({'sequence': sequence, 'label': int(0)}, ignore_index=True)
                #    df_final = df_final.append(df_neg)

    return df_final, df_target


if __name__ == '__main__':
    dataset = "msnbc"
    predict_data_type = "data_full/Zeroshot/"
    data_dir = "/data/prabhakar/CG/WNED/" + dataset + "/prediction_data/" + predict_data_type
    #predict_data_type = "testb/Cholan_AIDA/data_full/Zeroshot/"
    #data_dir = "/data/prabhakar/CG/CONLL-AIDA/prediction_data/" + predict_data_type

    df = pd.read_csv(data_dir + "ner_data.tsv", sep='\t', encoding='utf-8', usecols=['Sentence', 'Entity', 'Uri', 'WikiTitle','predictedEnt'])
    df_final, df_target = process_dataset(df, 5, False, dataset)
    df_final.to_csv(data_dir + "ned_data.tsv", index=False, sep="\t")
    df_target.to_csv(data_dir + "ned_target_data.tsv", index=False, sep="\t")
