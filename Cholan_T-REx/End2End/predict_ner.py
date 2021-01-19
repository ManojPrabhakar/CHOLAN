#from bert_ner import Ner
import pandas as pd

def predict_NER(model, df):
    #model = Ner("/data/prabhakar/manoj/code/NER/BERT-NER/pretrained_TREX_models/out_base_2/")
    final_entity_list = []
    predicted_entity_list = []

    #df = pd.read_csv("/data/prabhakar/CG/prediction_data/To_predict_5.tsv",  sep='\t', encoding='utf-8', usecols=['sequence1','sequence2Sep','uri','uriSequence2'])
    df = pd.DataFrame(df)
    sentence = df.sequence1.values

    for sent in sentence:
        output = model.predict(sent)
        predicted_entity_list.append(output)

    IFLAG = True

    for item, prediction in enumerate(predicted_entity_list):
        print("Item ", item)
        entity_list = []
        for i in range(0, len(prediction)):
            if prediction[i][1] == 'B-ENT':
                if prediction[-1][1] == 'B-ENT' or prediction[i + 1][1] != 'I-ENT':
                    B_word = prediction[i][0]
                    combined_word = B_word + ' EntityMentionSEP'
                    entity_list.append(combined_word)
                elif prediction[i + 1][1] == 'I-ENT':
                    if prediction[-1][1] == 'I-ENT':
                        B_word = prediction[i][0]
                        I_word = prediction[i + 1][0]
                        combined_word = B_word + ' ' + I_word + ' EntityMentionSEP'
                        entity_list.append(combined_word)
                    elif prediction[i + 2][1] == 'I-ENT':
                        B_word = prediction[i][0]
                        I_word = prediction[i + 1][0]
                        combined_word = B_word + ' ' + I_word
                        entity_list.append(combined_word)
                    elif prediction[i + 2][1] != 'I-ENT':
                        B_word = prediction[i][0]
                        I_word = prediction[i+1][0]
                        combined_word = B_word + ' ' + I_word + ' EntityMentionSEP'
                        entity_list.append(combined_word)

            elif prediction[i - 1][1] != 'B-ENT' and prediction[i][1] == 'I-ENT' and IFLAG == True:
                I_word = prediction[i][0]
                if prediction[i + 1][1] == 'I-ENT':
                    combined_word = I_word + ' '
                    entity_list.append(combined_word)
                else:
                    combined_word = I_word + ' EntityMentionSEP'
                    IFLAG = False
                    entity_list.append(combined_word)
            else:
                pass

        final_entity_list.append(' '.join(entity_list))

    df['predictedEnt'] = final_entity_list
    #df.to_csv("/data/prabhakar/CG/prediction_data/Predicted_5.tsv", index=False, sep="\t")

    return df
#sentence_input = input("Enter a sentence and get the tags\n")
#output = model.predict("Barack Obama is the husband of Michelle Obama.")
#print(final_entity_list)

