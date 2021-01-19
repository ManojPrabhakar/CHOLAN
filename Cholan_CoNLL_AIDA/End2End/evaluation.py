from post_processing import *
from utils import *

def EL_evaluation(target_qid_list, predicted_qid_list):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    target_ent_count = 0
    predicted_ent_count = 0
    matched_ent_count = 0
    match_list = []

    for i,t in enumerate(zip(target_qid_list, predicted_qid_list)):
        target_qids = t[0]
        predicted_qids = t[1]
        match = set(target_qids).intersection([i.strip() for i in predicted_qids])
        match_list.append(match)

    target_ent_count = sum([len(target_item) for target_item in target_qid_list])
    predicted_ent_count = sum([len(predicted_item) for predicted_item in predicted_qid_list])
    matched_ent_count = sum([len(matched_item) for matched_item in match_list])

    total_targets = target_ent_count
    total_predictions = predicted_ent_count
    total_matches = matched_ent_count
    if total_matches != 0:
        precision = (total_matches / total_predictions) * 100
    recall = (total_matches / total_targets) * 100
    fscore = 2 * ((precision * recall) / (precision + recall))

    print("\n### Evaluation Results ###")
    print("Total Targets = ",total_targets ,"Total predictions = ", total_predictions ,"Total matches = ", total_matches)
    print("Precision = %.1f" % precision, "\nRecall = %.1f" % recall, "\nF-Score = %.1f" % fscore)


def strict(labels, predictions):
	cnt = 0
	for label, pred in zip(labels, predictions):
		cnt += set(label) == set(pred)
	acc = cnt/len(labels)
	print("Strict Accuracy: %s" % acc)
	return acc


def strong_matching(target_list, predicted_list):
    predicted_count = 0
    target_count = 0
    matched_count = 0
    for target, predicted in zip(target_list, predicted_list):
        predicted_count += len(predicted)
        target_count += len(target)
        matched_count += len(set([t.strip() for t in target]).intersection(set([p.strip() for p in predicted])))

    micro_precision = matched_count/predicted_count
    micro_recall = matched_count/target_count
    micro_fscore = get_fscore(micro_precision, micro_recall)

    print("Total targets = ", target_count, "\tTotal predictions = ", predicted_count, "\tTotal matches = ", matched_count)
    return micro_precision*100, micro_recall*100, micro_fscore*100


def weak_macro(target_qid_list, predicted_qids_list):
    precision = 0
    recall = 0
    for target, predicted in zip(target_qid_list, predicted_qids_list):
        if len(predicted) > 0:
            precision += len(set(target).intersection(set([i.strip() for i in predicted])))/len(predicted)
        if len(target) > 0:
            recall += len(set(target).intersection(set([i.strip() for i in predicted])))/len(target)

    macro_precision = precision/len(target)
    macro_recall = recall/len(target)
    macro_fscore = get_fscore(macro_precision, macro_recall)
    return macro_precision*100, macro_recall*100, macro_fscore*100


def get_fscore(precision, recall):
    if (precision == 0) or (recall == 0):
        return 0
    return 2*(precision*recall)/(precision+recall)




if __name__ == '__main__':

    print("\n---Metrics Calculation---")
    #data_dir = "/data/prabhakar/CG/CONLL-AIDA/prediction_data/testa/Cholan_AIDA/data_500/"
    data_dir = "/data/prabhakar/CG/WNED/ace2004/prediction_data/data_full/"
    df_target = pd.read_csv(data_dir+"target_data_sent.tsv", sep='\t')
    df_predicted = pd.read_csv(data_dir+"predicted_data_sent.tsv", sep='\t')

    target_wikilabel_list, predicted_wikilabel_list = split_qids(df_target, df_predicted)

    #EL_evaluation(target_qid_list, predicted_qid_list)
    #macro_precision, macro_recall, macro_fscore = weak_macro(target_qid_list, predicted_qid_list)
    #print("\n---Macro Scores---")
    #print("Precision = %.1f" % macro_precision, "\tRecall = %.1f" % macro_recall, "\tF-Score = %.1f" % macro_fscore)

    print("\n--- Micro Scores - QID ---")
    micro_precision, micro_recall, micro_fscore = strong_matching(target_qids_list, predicted_qids_list)
    print("Precision = %.1f" % micro_precision, "\tRecall = %.1f" % micro_recall, "\tF-Score = %.1f" % micro_fscore)

    print("\n--- Micro Scores - WikiLabel ---")
    precision, recall, fscore = strong_matching(target_wikilabel_list, predicted_wikilabel_list)
    print("Precision = %.1f" % precision, "\tRecall = %.1f" % recall, "\tF-Score = %.1f" % fscore)
