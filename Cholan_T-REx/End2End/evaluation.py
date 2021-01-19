from utils import *
from post_processing import *

def EL_evaluation(target_qids_list, predicted_qids_list):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    match_list = []

    for i,t in enumerate(zip(target_qids_list, predicted_qids_list)):
        target_qids = t[0]
        predicted_qids = t[1]
        match = set(target_qids).intersection([i.strip() for i in predicted_qids])
        match_list.append(match)
        threshold = math.ceil(len(target_qids) / 2)

        if (len(match) >= (threshold - 1)):
            tp += 1
        elif (len(match) >= (threshold - 2)):
            fp += 1
        elif (len(predicted_qids) == 0):
            tn += 1
        else:
            fn += 1

    #match_set = set(match_list)
    match_set = set(tuple(x) for x in match_list)
    total_targets = 10000
    total_predictions = len(predicted_qids_list)
    total_matches = len(match_set)



    if total_matches != 0:
        precision = (total_matches / total_predictions) * 100
    recall = (total_matches / total_targets) * 100
    fscore = 2 * ((precision * recall) / (precision + recall))

    print("\n### Evaluation Results - 1 ###")
    print("Total Targets = ",total_targets ,"Total predictions = ", len(predicted_qids_list) ,"Total matches = ", total_matches)
    print("Precision = %.1f" % precision, "\nRecall = %.1f" % recall, "\nF-Score = %.1f" % fscore)

    print("\n### Evaluation Results - 2 ###")
    print("tp = ",tp, "fp = ",fp, "tn = ",tn, "fn = ",fn)
    return metrics(tp, fp, fn)



def metrics(tp, fp, fn):
    precision = (tp / (tp + fp)) * 100
    recall = (tp / (tp + fn)) * 100
    fscore = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, fscore



if __name__ == '__main__':
    data_dir = "/data/prabhakar/CG/prediction_data/data_10000/"
    df_target = pd.read_csv(data_dir+"target_data_sent.tsv", sep='\t')
    df_predicted = pd.read_csv(data_dir+"predicted_data_sent.tsv", sep='\t')

    target_list, predicted_list = split_qids(df_target, df_predicted)

    precision, recall, fscore = EL_evaluation(target_list, predicted_list)

    print("Precision = %.1f" % precision, "\nRecall = %.1f" % recall, "\nF-Score = %.1f" % fscore)