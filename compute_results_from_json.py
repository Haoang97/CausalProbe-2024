import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import find_first_digit, load_jsonlines

def calculate_metrics(data):
    total = len(data)
    correct = 0
    predicted = []
    gold = []
    invalid_ids = []

    for item in data:
        answer = str(item['golds'])
        output = item['output'].strip()
        chosen_id = find_first_digit(output)
        if chosen_id is not None:
            if int(answer) == chosen_id:
                correct += 1
            predicted.append(chosen_id)
            gold.append(int(answer))
        else:
            invalid_ids.append(item['id'])

    exact_match = correct / (total - len(invalid_ids))
    all_exact_match = correct / total
    precision = precision_score(gold, predicted, average='weighted')
    recall = recall_score(gold, predicted, average='weighted')
    f1 = f1_score(gold, predicted, average='weighted')

    return exact_match, all_exact_match, precision, recall, f1, invalid_ids

# read .json files
print("Please input a json foramt file that saves results:")
data = load_jsonlines(input())

# unify the keys
for item in data:
    if "golds" not in item:
        if "answers" in item:
            item["golds"] = item["answers"]
        if "answer" in item:
            item["golds"] = item["answer"]
        if "answerKey" in item:
            item["golds"] = [item["answerKey"]]
        if "label" in item:
            item["golds"] = item["label"]
        if "Correct_answer" in item:
            item["golds"] = item["Correct_answer"]
    if "id" not in item:
        if "index" in item:
            item["id"] = item["index"]

# compute metrics
exact_match, all_exact_match, precision, recall, f1, invalid_ids = calculate_metrics(data)

# print results
print(f"Exact Match: {exact_match:.4f}")
print(f"All Exact Match: {all_exact_match:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
