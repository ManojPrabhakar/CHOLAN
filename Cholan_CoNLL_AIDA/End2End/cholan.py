from bert_ner import Ner
from predict_ned import *
from predict_ner import predict_NER
from post_processing import *
from NED_Dataset_Creation import *
from test_model import *
from evaluation import strong_matching

dataset = "msnbc"
data_type = "Zeroshot/"
#predict_data_type = dataset + "/Cholan_AIDA/data_full/Zeroshot/"
#predict_data_dir = "/data/prabhakar/CG/CONLL-AIDA/prediction_data/" + predict_data_type
predict_data_type = "data_full/" + data_type
predict_data_dir = "/data/prabhakar/CG/WNED/" + dataset + "/prediction_data/" + predict_data_type

ned_model_dir = "/data/prabhakar/CG/NED_pretrained/model_data_50000/"
#ned_model_dir = "/data/prabhakar/CG/CONLL-AIDA/NED_pretrained/train_set_new/checkpoint-96761/"
#ned_model_dir = "/data/prabhakar/CG/CONLL-AIDA/NED_pretrained/0106/checkpoint-21933/"
ner_model_dir = "/data/prabhakar/manoj/code/NER/BERT-NER-CoNLL/pretrained_ner/"

def ner():
    # Load a trained NED_old model and vocabulary that you have fine-tuned
    ner_model = Ner(ner_model_dir)
    df = pd.read_csv(predict_data_dir + "To_predict.tsv", sep='\t', encoding='utf-8',
                     usecols=['Sentence', 'Entity', 'Uri', 'WikiTitle'])
    df = df.dropna()
    df_ner = predict_NER(ner_model, df)
    df_ner.to_csv(predict_data_dir + "ner_data.tsv", index=False, sep="\t")

    return df_ner


def ned(df_ned):
    # Load a trained NED_old model and vocabulary that you have fine-tuned
    ned_model = BertForSequenceClassification.from_pretrained(ned_model_dir)
    tokenizer = BertTokenizer.from_pretrained(ned_model_dir)
    prediction_dataloader, encoded_sequence = create_prediction_data(tokenizer, df_ned)

    predicted_labels, true_labels = test(ned_model, prediction_dataloader)
    df_predicted = pd.DataFrame(df_ned)
    df_predicted['predictedLabels'] = predicted_labels
    df_predicted_split, df_predicted_split_1, df_predicted_split_0 = process_sequence(df_predicted)

    df_predicted_split.to_csv(predict_data_dir + "predicted_data.tsv", index=False, sep="\t", columns=['EntityMention', 'Sentence', 'Predicted_Wikilabel', 'predictedLabels'])
    df_predicted_split_0.to_csv(predict_data_dir + "predicted_data_0.tsv", index=False, sep="\t", columns=['EntityMention', 'Sentence', 'Predicted_Wikilabel', 'predictedLabels'])
    df_predicted_split_1.to_csv(predict_data_dir + "predicted_data_1.tsv", index=False, sep="\t", columns=['EntityMention', 'Sentence', 'Predicted_Wikilabel', 'predictedLabels'])

    return df_predicted_split_1


if __name__ == '__main__':
    '''
    ### NER prediction ###
    df_ner = ner()
    
    ### Candidate Generation ###
    ### Format data to NED_old model format ###
    df_ner = pd.read_csv(predict_data_dir + "ner_data.tsv", sep='\t', encoding='utf-8',usecols=['Sentence', 'Entity', 'Uri', 'WikiTitle', 'predictedEnt'])
    df_ned, df_target = process_dataset(df_ner, 5, False, dataset)
    df_ned.to_csv(predict_data_dir + "ned_data.tsv", index=False, sep="\t")
    df_target.to_csv(predict_data_dir + "ned_target_data.tsv", index=False, sep="\t")
    '''

    ### NED Prediction ###
    df_ned = pd.read_csv(predict_data_dir + "ned_data.tsv", sep='\t', encoding='utf-8', usecols=['sequence1', 'sequence2', 'label'])
    df_predicted = ned(df_ned)


    ### Evaluation ###
    df_target = pd.read_csv(predict_data_dir + "ned_target_data.tsv", sep="\t", encoding='utf-8')
    df_predicted = pd.read_csv(predict_data_dir + "predicted_data_1.tsv", sep="\t", encoding='utf-8')
    #target_qids_list, predicted_qids_list, target_wikilabel_list, predicted_wikilabel_list = combine_sentence(df_target, df_predicted, predict_data_dir)
    target_wikilabel_list, predicted_wikilabel_list = combine_sentence(df_target, df_predicted, predict_data_dir)

    print("### Evaluation Results ###")

    #precision, recall, fscore = strong_matching(target_qids_list, predicted_qids_list)
    #print("--- Micro Scores - QID ---")
    #print("Precision = %.1f" % precision, "\tRecall = %.1f" % recall, "\tF-Score = %.1f" % fscore)

    precision, recall, fscore = strong_matching(target_wikilabel_list, predicted_wikilabel_list)
    print("--- Micro Scores - WikiLabel ---")
    print("Precision = %.1f" % precision, "\tRecall = %.1f" % recall, "\tF-Score = %.1f" % fscore)






