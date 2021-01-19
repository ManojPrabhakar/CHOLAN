import pandas as pd
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

from candidate_generation import *


def process_dataset(df, size):
    # Load the dataset into a pandas dataframe.
    #df = pd.read_csv("/data/prabhakar/CG/Trex_data/Trex_train_50000.tsv",  sep='\t', encoding='utf-8', usecols=['sequence1','sequence2Sep','uri','uriSequence2'])
    df = df.dropna()
    df = df.rename(columns={"sequence1": "Sentence", "predictedEnt": "EntitySep", "uri": "Qid", "uriSequence2": "WikiLabel", "sequence2Sep":"TargetEnt"})
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

    ent = 0

    for i in range(0, df.shape[0]):
        for j in range(0,len(Predicted_EntityMentions[i])-1):
            ent += 1
            print('Sentence - ', i+1, ', Entity - ', j+1, ', Total_Entity - ', ent)
#            POS_FLAG = False
            CG_query = Predicted_EntityMentions[i][j]
            CG_result = entity_search(CG_query, size)
            CG_result_set = set(tuple(x) for x in CG_result)

            df_target_intermediate = pd.DataFrame()
            seq1 = Predicted_EntityMentions[i][j] + ' | ' + df.iloc[i]['Sentence']
            #target_seq2 = Target_WikiLabels[i][j] + ' | ' + Target_Qids[i][j]
            #target_sequence = seq1 + ' | ' + target_seq2
            if j < len(Target_EntityMentions[i])-1:
                df_tgt = df_target_intermediate.append({'EntityMention': Target_EntityMentions[i][j] , 'Sentence': df.iloc[i]['Sentence'], 'Target_Qid': Target_Qids[i][j], 'Target_Wikilabel': Target_WikiLabels[i][j], 'label': int(1)}, ignore_index=True)
                df_target = df_target.append(df_tgt)

            for r, result in enumerate(CG_result_set):
                df_intermediate = pd.DataFrame(columns=['sequence', 'label'])
                result_wikilabel = result[0]
                result_qid = result[1].strip("<http://www.wikidata.org/entity/>")

                #if result_qid
                pred_seq2 = result_wikilabel + ' | ' + result_qid
                pred_sequence = seq1 + ' | ' + pred_seq2
                df_CG = df_intermediate.append({'sequence': pred_sequence, 'label': int(0)}, ignore_index=True)
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
    data_dir = "/data/prabhakar/CG/prediction_data/data_10000/"
    df = pd.read_csv(data_dir + "ner_data.tsv", sep='\t', encoding='utf-8', usecols=['sequence1', 'sequence2Sep', 'uri', 'uriSequence2','predictedEnt'])
    df_final, df_target = process_dataset(df, 30)
    df_final.to_csv(data_dir + "ned_data.tsv", index=False, sep="\t")
    df_target.to_csv(data_dir + "ned_target_data.tsv", index=False, sep="\t")
