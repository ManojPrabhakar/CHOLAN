from bert_ner import Ner
import pandas as pd

def predict_NER(model, df):
    final_entity_list = []
    predicted_entity_list = []

    #df = pd.DataFrame(df)
    df = df.fillna('')
    sentence = df.Sentence.values
    target_entity_list = [row['Entity'].split('EntityMentionSEP') for index, row in df.iterrows()]

    for sent in sentence:
        output = model.predict(sent)
        predicted_entity_list.append(output)

    IFLAG = True

    for item, prediction in enumerate(predicted_entity_list):
        print("Item ", item)
        entity_list = []
        for i in range(0, len(prediction)):
            if prediction[i][1][0] == 'B':
                if prediction[-1][1][0] == 'B' or prediction[i + 1][1][0] != 'I':
                    B_word = prediction[i][0]
                    combined_word = B_word + ' EntityMentionSEP'
                    entity_list.append(combined_word)
                elif prediction[i + 1][1][0] == 'I':
                    if prediction[-1][1][0] == 'I':
                        B_word = prediction[i][0]
                        I_word = prediction[i + 1][0]
                        combined_word = B_word + ' ' + I_word + ' EntityMentionSEP'
                        entity_list.append(combined_word)
                    elif prediction[i + 2][1][0] == 'I':
                        B_word = prediction[i][0]
                        I_word = prediction[i + 1][0]
                        combined_word = B_word + ' ' + I_word
                        entity_list.append(combined_word)
                    elif prediction[i + 2][1][0] != 'I':
                        B_word = prediction[i][0]
                        I_word = prediction[i+1][0]
                        combined_word = B_word + ' ' + I_word + ' EntityMentionSEP'
                        entity_list.append(combined_word)

            elif prediction[i - 1][1][0] != 'B' and prediction[i][1][0] == 'I' and IFLAG == True:
                I_word = prediction[i][0]
                if prediction[i + 1][1][0] == 'I':
                    combined_word = I_word + ' '
                    entity_list.append(combined_word)
                else:
                    combined_word = I_word + ' EntityMentionSEP'
                    IFLAG = False
                    entity_list.append(combined_word)
            else:
                pass

        #check_entity_list = check_entity(target_entity_list[item], entity_list)
        final_entity_list.append(' '.join(entity_list))

    df['predictedEnt'] = final_entity_list

    return df

def check_entity(target_list, predicted_list):
    check_entity_list = []
    predicted = [e.split('EntityMentionSEP') for e in predicted_list]
    #
    check_entity_list = target_list
    return check_entity_list

if __name__ == '__main__':
    dataset = "ace2004"
    # predict_data_type = dataset + "/Cholan_AIDA/data_full/Indexed"
    # predict_data_dir = "/data/prabhakar/CG/CONLL-AIDA/prediction_data/" + predict_data_type + "/"
    predict_data_type = "data_full/Indexed/"
    predict_data_dir = "/data/prabhakar/CG/WNED/" + dataset + "/prediction_data/" + predict_data_type

    ner_model_dir = "/data/prabhakar/manoj/code/NER/BERT-NER-CoNLL/pretrained_ner/"

    ner_model = Ner(ner_model_dir)
    df = pd.read_csv(predict_data_dir + "To_predict_error.tsv",  sep='\t', encoding='utf-8', usecols=['Sentence','Entity','Uri','WikiTitle'])

    df_result = predict_NER(ner_model, df)

    df_result.to_csv(data_dir + "ner_data_error.tsv", index=False, sep="\t")

    # sentence_input = input("Enter a sentence and get the tags\n")
    # output = model.predict("Barack Obama is the husband of Michelle Obama.")
    # print(final_entity_list)
