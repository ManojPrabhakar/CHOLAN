from utils import *

def process_sequence(df_predicted):
    ## Split sequence into columns ##
    df_predicted['EntityMention'] = df_predicted['sequence1'].apply(lambda seq: split_sequence(seq, index=0))
    df_predicted['Sentence'] = df_predicted['sequence1'].apply(lambda seq: split_sequence(seq, index=1))
    df_predicted['Predicted_Wikilabel'] = df_predicted['sequence2'].apply(lambda seq: split_sequence(seq, index=0))
    #df_predicted['Predicted_Qid'] = df_predicted['sequence2'].apply(lambda seq: split_sequence(seq, index=1))
    #df_predicted['Sentence'] = df_predicted['sequence2'].apply(lambda seq: split_sequence(seq, index=2))

    ## Split the 0 & 1 predictions ##
    df_predicted_0 = df_predicted[df_predicted['predictedLabels'] == 0]
    df_predicted_1 = df_predicted[df_predicted['predictedLabels'] == 1]

    df_predicted_1 = df_predicted_1.drop_duplicates(subset=['EntityMention'], keep='first')

    return df_predicted, df_predicted_1, df_predicted_0


def split_sequence(seq, index):
    seq_list = str(seq).split('|')
    return seq_list[index]


def combine_sentence(df_target, df_predicted, predict_data_dir):
    ## Combine multiple sentence instances into one ##
    df_predicted_sent = df_predicted.groupby('Sentence', as_index=False).agg({'EntityMention':'|'.join, 'Predicted_Wikilabel':'|'.join})
    df_target_sent = df_target.groupby('Sentence', as_index=False).agg({'EntityMention':'|'.join, 'Target_Wikilabel':'|'.join})

    ## Get target sentences for the corresponding predictions ##
    df_final_target = df_target_sent[((df_target_sent['Sentence'].str.strip()).isin(df_predicted_sent['Sentence'].str.strip()))]

    ## Sorting the dataframes for evaluation ##
    df_final_target_sorted = sort_sentences(df_final_target, 'Sentence')
    df_predicted_sent_sorted = sort_sentences(df_predicted_sent, 'Sentence')

    df_predicted_sent_sorted.to_csv(predict_data_dir + "predicted_data_sent.tsv", index=False, sep='\t',
                                    columns=['EntityMention', 'Sentence', 'Predicted_Wikilabel'])
    df_final_target_sorted.to_csv(predict_data_dir + "target_data_sent.tsv", index=False, sep='\t',
                                  columns=['EntityMention', 'Sentence', 'Target_Wikilabel', 'Target_Qid'])
    return split_qids(df_final_target_sorted, df_predicted_sent_sorted)


def sort_sentences(df_file, column):
    return df_file.sort_values([column])

def split_qids(df_final, df_predicted):
    #target_qids_list = [row['Target_Qid'].split('|') for index, row in df_final.iterrows()]
    #predicted_qids_list = [row['Predicted_Qid'].split('|') for index, row in df_predicted.iterrows()]
    target_wikilabel_list = [row['Target_Wikilabel'].split('|') for index, row in df_final.iterrows()]
    predicted_wikilabel_list = [row['Predicted_Wikilabel'].split('|') for index, row in df_predicted.iterrows()]
    #return target_qids_list, predicted_qids_list, target_wikilabel_list, predicted_wikilabel_list
    return target_wikilabel_list, predicted_wikilabel_list