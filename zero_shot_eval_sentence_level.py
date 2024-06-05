import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import openai
import os
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import re

def save_to_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

def read_from_json(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    return data

def zero_shot_evaluation(sentence_label_dicts,model_name,document_name):
  output_sentence_label_dict_list = []
  sentence_label_dicts = [element for element in sentence_label_dicts if element['doc_name'] == document_name] #use chosen document name.
  for item in tqdm(sentence_label_dicts, desc="Processing Sentences"):
    if item['doc_name'] == document_name:
        sentence = item['sentence']
        if(len(sentence) > 3):
          # sentence len > 3 chosen such that punctuation marks are avoided.
          chat_completion = client.chat.completions.create(
          messages=[
              {
                  "role": "user",
                  "content": "Given a sentence of Indian Legal Court Case document, classify it into one of the following labels: {Facts, Ruling by Lower Court, Argument, Statute, Precedent, Ratio of the decision, Ruling by Present Court}. Make sure the output is strictly in the same format as the given set of labels without any extra words or artifacts. Sentence: "+sentence,
              }
          ],
            model=model_name,
          )
          item['pred_label'] = chat_completion.choices[0].message.content
          output_sentence_label_dict_list.append(item)


  return output_sentence_label_dict_list

###### Execution ########

api_key = input("Please provide an OpenAI API key:\n")
openai.api_key = api_key

from openai import OpenAI

client = OpenAI(
    api_key=api_key,
)

#read data
paheli_sentence_label_dicts = read_from_json('paheli_sentence_label_dicts.json')

#model_name = "gpt-3.5-turbo"
model_name = "gpt-4-turbo" #change model name to desired gpt model.
paheli_output_predictions = zero_shot_evaluation(paheli_sentence_label_dicts, model_name,'2009_S_146.txt') # Document name to perform prediction on
prediction_output_file_name = model_name + "paheli_output_predictions.json"

# Check if the output file exists
if os.path.exists(prediction_output_file_name):
    # If the file exists, read its content
    with open(prediction_output_file_name, 'r') as f:
        existing_data = json.load(f)

    # Append the new predictions to the existing data
    if isinstance(existing_data, list):
        existing_data.extend(paheli_output_predictions)
    else:
        existing_data = paheli_output_predictions  # If the existing data is not a list, overwrite it

    # Save the updated data back to the file
    save_to_json(existing_data, prediction_output_file_name)
else:
    # If the file does not exist, create it and save the new predictions
    save_to_json(paheli_output_predictions, prediction_output_file_name)
