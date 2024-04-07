from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import json
import numpy as np

lan = 'java.jsonl'
train = 'javatrain.jsonl'
output_filename = 'java_with_best.jsonl'

def compute_bm25_top_10_scores(jsonl_filename, query_string):
    # Read the JSONL file and initialize a list to store the JSON entries
    entries = []
    with open(jsonl_filename, 'r') as file:
        for line in file:
            entry = json.loads(line)
            entries.append(entry)

    # Tokenize the query string
    tokenized_query = query_string.lower().split(" ")

    # Initialize BM25 with tokenized corpus from all 'diff' entries
    tokenized_corpus = [word_tokenize(entry['diff'].lower()) for entry in entries]
    bm25 = BM25Okapi(tokenized_corpus)

    # Compute BM25 scores for each 'diff' entry
    doc_scores = bm25.get_scores(tokenized_query)

    # Find the indices of the entries with the top 10 BM25 scores
    top_10_indices = np.argsort(doc_scores)[::-1][:10]

    # Retrieve the 'diff', 'msg', and 'diff_id' for each of the top 10 entries
    top_10_matches = [{'diff': entries[i]['diff'], 'msg': entries[i]['msg'], 'diff_id': entries[i]['diff_id']} for i in top_10_indices]

    return top_10_matches

# Open the original JSONL file and read the data
with open(lan, 'r', encoding='utf8') as f:
    json_data = f.readlines()

# Create a new JSONL file to store the results


# Iterate through the JSON data and compute BM25 top 10 scores
for item in json_data:
    data = json.loads(item)
    top_10_matches = compute_bm25_top_10_scores(train, data['diff'])

    # Create a new dictionary with the original diff_id and the new fields for top 10 matches
    modified_entry = {'diff_id': data['diff_id']}
    for i, match in enumerate(top_10_matches, 1):
        modified_entry[f'best_diff{i}'] = match['diff']
        modified_entry[f'best_msg{i}'] = match['msg']
        modified_entry[f'best_id{i}'] = match['diff_id']

    # Write the modified entry to the output file
    with open(output_filename, 'a', encoding='utf8') as output_file:
        output_file.write(json.dumps(modified_entry) + '\n')

    # Optional: Print the message of the top match for each 'diff_id'
    print(f'Top match for diff_id {data["diff_id"]}: {top_10_matches[0]["msg"]}')

# Indicate completion
print("Results have been saved to", output_filename)

