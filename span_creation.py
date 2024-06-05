import os
import json

def save_to_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

def read_from_json(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    return data

def collapse_sentences(sentences):
    if not sentences:
        return []

    # Step 1: Sort by doc_name and id
    sentences.sort(key=lambda x: (x['doc_name'], x['id']))

    collapsed_sentences = []
    current_entry = None

    for sentence in sentences:
        if current_entry is None:
            current_entry = sentence
        elif (current_entry['doc_name'] == sentence['doc_name'] and
              current_entry['label'] == sentence['label']):
            current_entry['sentence'] += " " + sentence['sentence']
        else:
            collapsed_sentences.append(current_entry)
            current_entry = sentence

    if current_entry is not None:
        collapsed_sentences.append(current_entry)

    return collapsed_sentences

### Read processed data
paheli_sentence_label_dicts = read_from_json('paheli_sentence_label_dicts.json')

### Create Spans and Save
paheli_spans_label_dicts = collapse_sentences(paheli_sentence_label_dicts)
save_to_json(paheli_spans_label_dicts,'paheli_spans_label_dicts.json')
