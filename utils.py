import jsonlines
import json
import copy
import re

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst
    
def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def save_file_json(data, fp):
    with open(fp, "w") as file:
        json.dump(data, file)

def process_arc_instruction(item, instruction):
    choices = item["choices"]
    answer_labels = {}
    for i in range(len(choices["label"])):
        answer_key = choices["label"][i]
        text = choices["text"][i]
        if answer_key == "1":
            answer_labels["A"] = text
        if answer_key == "2":
            answer_labels["B"] = text
        if answer_key == "3":
            answer_labels["C"] = text
        if answer_key == "4":
            answer_labels["D"] = text
        if answer_key in ["A", "B", "C", "D"]:
            answer_labels[answer_key] = text

    if "D" not in answer_labels:
        answer_labels["D"] = ""
    choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
    if "E" in answer_labels:
        choices += "\nE: {}".format(answer_labels["E"])
    processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
    return processed_instruction

def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output

def clean_dict_list(input_data):
    """
    清洗包含None值的字典列表。
    
    Args:
        input_data (list): 包含字典的列表。
        
    Returns:
        list: 清洗后的字典列表。
    """
    cleaned_data = []
    for d in input_data:
        # 检查每个字典是否包含None值
        if all(value is not None for value in d.values()):
            cleaned_data.append(d)
    return cleaned_data

def find_first_boolean_batch(strings):
    # 用于保存每个字符串中第一次出现的"true"或"false"
    results = []

    # 正则表达式，用于匹配"true"或"false"
    pattern = re.compile(r"\b(true|false)\b")

    for string in strings:
        # 搜索当前字符串中的"true"或"false"
        match = pattern.search(string)
        if match:
            # 如果找到，添加到结果列表
            results.append(match.group())
        else:
            # 如果没有找到，可以根据需要添加None或者保持不添加
            results.append(None)

    return results

def find_first_boolean(string):
    # 正则表达式，用于匹配"true"或"false"
    pattern = re.compile(r"\b(true|false)\b", re.IGNORECASE)

    # 搜索当前字符串中的"true"或"false"
    match = pattern.search(string)
    if match:
        # 如果找到，返回匹配的单词
        return match.group()
    else:
        # 如果没有找到，返回None
        return None
        
def string_to_boolean(s):
    # 将字符串转换为小写，以便不区分大小写进行比较
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        # 如果输入不是"true"或"false"，可以返回None或者抛出异常
        return None

def find_first_digit(string):
    """
    Find the first digit in a given string and return it as an integer.
    If no digit is found, return None.
    """
    for char in string:
        if char.isdigit():
            return int(char)
    return None