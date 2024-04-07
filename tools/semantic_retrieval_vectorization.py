from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import json
# 准备您的代码差异文本
from nltk.tokenize import word_tokenize
import json
import re
import nltk
import traceback
from openai import OpenAI
import time

lan = 'py1.jsonl'
output_file = 'vpy1no.jsonl'

def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s

def to_Underline(x):
    """转空格命名"""
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', ' \g<0>', x).lower()

def remove_between_identifiers(text, identifier_start, identifier_end):
    # 定义正则表达式模式
    pattern = f'(?<={identifier_start}).*?(?={identifier_end})'

    # 使用re.sub方法替换匹配到的部分为空字符串
    result = re.sub(pattern, '', text)
    if identifier_start == 'mmm a':
        result = result.replace('mmm a<nl>', '')
    if identifier_start == 'ppp b':
        result = result.replace('ppp b<nl>', '')
        result = result.replace('<nl>', '\n')
    result = result.replace(' . ', '.')
    result = result.replace('  ', '.')
    result = result.replace(' = ', '=')
    result = result.replace(' ; ', ';')
    result = result.replace(' (', '(')
    result = result.replace(') ', ')')
    return result
def get_tokens(text):
    if len(text) > 510:
        return ' '.join(text[:510])
    else:
        return ' '.join(text)
def process_diff(diff):
    wordsGPT = diff.split()
    msgGPT_list = []
    for wordGPT in wordsGPT:
        if len(wordGPT) > 1:
            if is_camel_case(wordGPT):
                msgGPT_list.append(to_Underline(wordGPT))
            else:
                msgGPT_list.append(wordGPT)
        else:
            msgGPT_list.append(wordGPT)
    diff = ' '.join(msgGPT_list)

    result = remove_between_identifiers(diff, 'mmm a', '<nl>')
    diff = remove_between_identifiers(result, 'ppp b', '<nl>')

    return get_tokens(diff)
def preprocess_code_diff(diff_text):
    """
    Preprocesses a code difference text by replacing lines starting with "+" with "[ADD]",
    lines starting with "-" with "[DEL]", and adding "[KEEP]" at the beginning of lines that
    do not start with "+" or "-".

    Parameters:
    diff_text (str): The original code difference text.

    Returns:
    str: The preprocessed code difference text.
    """
    # Split the text into individual lines
    lines = diff_text.split('\n')

    # Preprocess each line
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('+'):
            processed_lines.append('[ADD] ' + stripped_line[1:].strip())
        elif stripped_line.startswith('-'):
            processed_lines.append('[DEL] ' + stripped_line[1:].strip())
        else:
            processed_lines.append('[KEEP] ' + stripped_line)

    # Combine the processed lines back into a single string
    processed_diff = '\n'.join(processed_lines)

    return processed_diff


# 选择适当的预训练模型和tokenizer
model_name = "microsoft/codereviewer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 移动模型到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 读取JSON Lines文件并对每个diff进行向量化
with open(lan, 'r',encoding='utf8') as file, open(output_file, 'a') as outfile:
    for line in file:

        json_line = json.loads(line)

        if 'diff' in json_line and 'diff_id' in json_line:
            # 获取diff字段和diff_id
            diff = json_line['diff']
            result = remove_between_identifiers(diff, 'mmm a', '<nl>')
            code_diff1 = get_tokens(remove_between_identifiers(result, 'ppp b', '<nl>'))
            code_diff = preprocess_code_diff(code_diff1)
            diff_id = json_line['diff_id']

            # 对代码差异文本进行tokenization，并将其移动到 GPU 上
            input_ids = tokenizer.encode(code_diff, truncation=True, max_length=510, return_tensors="pt").to(device)

            # 获取编码向量
            model.eval()
            with torch.no_grad():
                encoder_outputs = model.encoder(input_ids)
            encoded_vector = encoder_outputs.last_hidden_state.cpu().numpy()
            # 获取第一个输出作为编码向量
            cls_vector = encoder_outputs.last_hidden_state[0, 0, :].cpu().numpy().tolist()

            # cls_vector现在是一个包含768个元素的一维数组
            #print(len(cls_vector))
            # 创建一个包含diff_id和编码向量的字典
            output_dict = {
                "diff_id": diff_id,
                "cls_vector": cls_vector
            }

            # 将字典写入输出文件
            outfile.write(json.dumps(output_dict) + '\n')
