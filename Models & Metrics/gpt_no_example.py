from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import json
import re
import numpy as np
import nltk
import traceback
from openai import OpenAI
import time
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
    tokens = nltk.word_tokenize(text)
    if len(tokens) > 1024:
        return ' '.join(tokens[:1024])
    else:
        return ' '.join(tokens)
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


# 打开JSONL文件并读取数据
with open('py1.jsonl', 'r',encoding='utf8') as f:
    json_data = f.readlines()
data = {"diff_id": 0, "msg": f"0", "msgGPT": f"0", "METEOR Score": f"0", "BLEU Score": f"0", "ROUGE-L Score": f"0"}

# 遍历 JSON 数据，提取并存储 diff 和 msg
num = 0
temp = 0
for item in json_data:
    attempts = 0
    while attempts < 5:
        key_list = [
            "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",
            "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",
            "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",
            "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",
            "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",

        ]
        key = key_list[temp]
        client = OpenAI(
            api_key=key,
        )
        temp += 1
        if temp == 5:
            temp = 0
        # 解析 JSON 数据
        data = json.loads(item)

        # 提取 diff 和 msg
        diff_id = data['diff_id']
        diff = data['diff']
        result = remove_between_identifiers(diff, 'mmm a', '<nl>')
        diff = get_tokens(remove_between_identifiers(result, 'ppp b', '<nl>'))
        msg = data['msg']
        words = msg.split()
        msg_list = []
        for word in words:
            if len(word) > 1:
                if is_camel_case(word):
                    msg_list.append(to_Underline(word))
                else:
                    msg_list.append(word)
            else:
                msg_list.append(word)
        msg = ' '.join(msg_list)
        # Example usage:
        if num < 4:
            num += 1
        elif num >= 4:
            num = 0
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a programmer who makes the above code changes."},
                    {"role": "user",
                     "content": f'''{diff}\n Please write a commit message that contains only one simple sentence for the above code change.\n'''},
                    # {"role": "user", "content": "Use END_OF_CASE to finish your answer.\n"},
                ],
                max_tokens=50,
                temperature=0.8,
                n=30,
                top_p=0.95
            )
            num_answers = 30
            msgGPTs = []
            for i in range(num_answers):
                msgGPT = completion.choices[i].message.content
                wordsGPT = msgGPT.split()
                msgGPT_list = []
                for wordGPT in wordsGPT:
                    if len(wordGPT) > 1:
                        if is_camel_case(wordGPT):
                            msgGPT_list.append(to_Underline(wordGPT))
                        else:
                            msgGPT_list.append(wordGPT)
                    else:
                        msgGPT_list.append(wordGPT)
                msgGPT = ' '.join(msgGPT_list)
                msgGPTs.append(msgGPT)
            print(msgGPTs)

            # 将 diff 和 msg ,score添加到列表中
            data = {"diff_id": diff_id, "msg": f"{msg}"}
            for i in range(30):
                data[f"msgGPT{i}"] = f"{msgGPTs[i]}"
            with open('pygptnoexample.jsonl', 'a') as f:
                json.dump(data, f)
                f.write('\n')
            break
        except:
            traceback.print_exc()
            data = {"diff_id": diff_id, "msg": f"0", "msgGPT": f"0"}
            with open('pygptnoexample.jsonl', 'a') as f:
                json.dump(data, f)
                f.write('\n')
            time.sleep(1)
            attempts += 1
            if attempts == 5:
                print(f"{item} 已经重试了3次，仍然失败。")
                # 这里可以选择记录失败的item，或者是进行其他错误处理
                # ...
                break  # 重试达到3次后，跳出内部循环，处理下一个item
