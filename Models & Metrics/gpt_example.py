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
    if len(tokens) > 600:
        return ' '.join(tokens[:600])
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


output_filenames = {
    1: 'javaadd16k_gpt_1_noselect.jsonl',
3: 'javaadd16k_gpt_3_noselect.jsonl',
5: 'javaadd16k_gpt_5_noselect.jsonl',
10: 'javaadd16k_gpt_10_noselect.jsonl',
}

# 打开JSONL文件并读取数据
with open('java2.jsonl', 'r',encoding='utf8') as f:
    json_data = f.readlines()
data = {"diff_id": 0, "msg": f"0", "msgGPT": f"0", "METEOR Score": f"0", "BLEU Score": f"0", "ROUGE-L Score": f"0"}

# 遍历 JSON 数据，提取并存储 diff 和 msg
num = 0
temp = 0
for item in json_data:
    attempts = 0
    while attempts < 5:
        key_list = [
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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',
         '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'
        ]
        key = key_list[temp]
        client = OpenAI(
            api_key=key,
        )
        temp += 1
        if temp == 20:
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
        # 提取对应的best_diff和best_msg
        best_diffs_msgs = []
        with open('javabest_no_select.jsonl', 'r',encoding='utf8') as file:
            for line in file:
                best_data = json.loads(line)
                if best_data['diff_id'] == diff_id:
                    for i in range(1, 11):
                        diff_key = f'best_diff{i}'
                        msg_key = f'best_msg{i}'
                        if diff_key in best_data and msg_key in best_data:
                            # 应用相同的预处理步骤
                            result_b = remove_between_identifiers(best_data[diff_key], 'mmm a', '<nl>')
                            best_diff = get_tokens(remove_between_identifiers(result_b, 'ppp b', '<nl>'))
                            best_msg = best_data[msg_key]
                            best_diffs_msgs.append((best_diff, best_msg))
                    break

        if num < 4:
            num += 1
        elif num >= 4:
            num = 0
        try:
            for num_examples in [1,3,5,10]:
                if len(best_diffs_msgs) >= num_examples:
                    # 构建prompt
                    prompt = ""
                    for best_diff, best_msg in best_diffs_msgs[:num_examples]:
                        prompt += f"{best_diff}\nPlease write a commit message for the above code change.\n{best_msg}\n\n"
                    prompt += f"{diff}\nPlease write a commit message for the above code change.\n"

                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo-16k",
                        messages=[
                            {"role": "system", "content": "You are a programmer who makes the above code changes."},
                            {"role": "user",
                             "content": f'''{prompt}'''}
                        ],
                        max_tokens=50,
                        temperature=0.8,
                        n=5,
                        top_p=0.95
                    )
                    num_answers = 5
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
                    for i in range(5):
                        data[f"msgGPT{i}"] = f"{msgGPTs[i]}"
                        output_data = {
                            "diff_id": diff_id,
                            "msg": msg
                        }

                        for i in range(5):
                            output_data[f"msgGPT{i}"] = msgGPTs[i]
                    with open(output_filenames[num_examples], 'a', encoding='utf8') as f:
                        json.dump(output_data, f)
                        f.write('\n')
            break
        except:
            traceback.print_exc()
            time.sleep(1)
            attempts += 1
            if attempts == 5:
                print(f"{item} 已经重试了3次，仍然失败。")
                # 这里可以选择记录失败的item，或者是进行其他错误处理
                # ...
                break  # 重试达到3次后，跳出内部循环，处理下一个item
