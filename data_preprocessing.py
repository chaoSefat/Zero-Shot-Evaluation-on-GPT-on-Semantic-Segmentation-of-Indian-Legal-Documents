import os
import json

  def process_text_files(directory_path):
    sentences = []
    labels = []
    sentence_label_dicts = []

    # Auto incrementing ID
    auto_id = 1

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    # Split line into sentence and label
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        sentence, label = parts
                        sentences.append(sentence)
                        labels.append(label)
                        sentence_label_dicts.append({
                            'id': auto_id,
                            'doc_name': filename,
                            'sentence': sentence,
                            'label': label
                        })
                        auto_id += 1

    return sentences, labels, sentence_label_dicts

def save_to_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

def read_from_json(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    return data

directory_path = #path to the dataset text files
sentences, labels, sentence_label_dicts = process_text_files(directory_path)
### Saving the processed data as Json
output_json_file = 'paheli_sentence_label_dicts.json'
save_to_json(sentence_label_dicts, output_json_file)
