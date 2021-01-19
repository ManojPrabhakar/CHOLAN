import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, confusion_matrix
from utils import *
from cholan import predict_data_dir

# Prediction on test set
def test(model, test_dataloader):
    print('Predicting labels for {:,} test sentences...'.format(len(test_dataloader)))

    if torch.cuda.is_available():
        #torch.cuda.set_device(2)
        device = torch.device("cuda")
        print('GPU available -', device)
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Put model in evaluation mode
    model.eval()
    model.cuda()
    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in tqdm(test_dataloader, desc="Evaluation"):
        #batch = tuple(t.to(device) for t in batch)

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_token_type_ids = batch[3].to(device)

#        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating Matthews Corr. Coef. for each batch...')

    matthews_set = []
    # For each input batch...
    for i in range(len(true_labels)):
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        #print(pred_labels_i)

        # Calculate and store the coef for this batch.
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)

    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    print('MCC: %.3f' % mcc)

    matrix_results, accuracy, report = compute_metrics(flat_predictions, flat_true_labels)
    # results.update(result)

    output_eval_file = os.path.join(predict_data_dir, "Prediction_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test results {} *****")
        logger.info("Confusion Matrix:\n %s", str(matrix_results))
        logger.info(" Accuracy: %s", str(accuracy))
        logger.info(" MCC: %.3f", mcc)
        logger.info(" Classification Report:\n%s", str(report))
        writer.write("Confusion Matrix:\n %s" % str(matrix_results))
        writer.write(" Accuracy: %s" % str(accuracy))
        writer.write(" MCC: %.3f" % mcc)
        writer.write(" Classification Report:\n%s" % str(report))

    return flat_predictions, flat_true_labels
