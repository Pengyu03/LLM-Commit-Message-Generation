import json
import os
import openai
import re
import time
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import traceback
import nltk
from gradio_client import Client
#from tenacity import (
#    retry,
 #   retry_if_exception_type,
  #  wait_random_exponential,
#)
#@retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError)))
def calculate_meteor(sentence1, sentence2):
    """
    计算两个句子之间的METEOR分数
    """
    # 将两个句子转换为词频向量
    vectorizer = CountVectorizer().fit([sentence1, sentence2])
    sentence1_vector = vectorizer.transform([sentence1])
    sentence2_vector = vectorizer.transform([sentence2])
    
    # 计算两个向量的余弦相似度
    similarity = cosine_similarity(sentence1_vector, sentence2_vector)[0][0]
    
    # 根据METEOR公式计算分数
    score = 2 * similarity * len(sentence1) * len(sentence2) / (len(sentence1) + len(sentence2))
    return score

def calculate_bleu(reference, translation):
    """
    计算BLEU分数
    """
    bleu_score = sentence_bleu([reference], translation)
    return bleu_score

def calculate_rouge_l(reference, translation):
    """
    计算ROUGE-L分数
    """
    rouge = Rouge()
    rouge_l_score = rouge.get_scores(translation, reference, avg=True)['rouge-l']
    return rouge_l_score

def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def to_Underline(x):
    """转空格命名"""
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', ' \g<0>', x).lower()

def get_tokens(text):
    tokens = nltk.word_tokenize(text)
    if len(tokens) > 1024:
        return ' '.join(tokens[:1024])
    else:
        return ' '.join(tokens)

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

# 打开JSONL文件并读取数据
with open('output999.jsonl', 'r',encoding='UTF-8') as f:
    json_data = f.readlines()
data = {"diff_id":0, "msg": f"0", "msgGPT": f"0", "METEOR Score" : f"0", "BLEU Score" : f"0","ROUGE-L Score":f"0"}
with open('java_final_no_result_7B7.jsonl', 'a',encoding='UTF-8') as f:
    json.dump(data, f)
    f.write('\n')
# 遍历 JSON 数据，提取并存储 diff 和 msg
#num = 0
for item in json_data:
    attempts = 0
    while attempts < 3:

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

       #if num < 4:
        #    num += 1
        #elif num >= 4:
         #   num = 0
        client = Client("huggingface-projects/llama-2-13b-chat")
        try:
            response = client.predict(
                f'''
    {diff}\n Please write a commit message that contains only one simple sentence for the above code change.\n''',
                # {"role": "user", "content": "Use END_OF_CASE to finish your answer.\n"},
                "You are a programmer who makes the above code changes.",  # 替换为您的消息文本
                1024,  # 替换为您的参数值
                0.1,  # 替换为您的参数值
                0.05,  # 替换为您的参数值
                1,  # 替换为您的参数值
                1,  # 替换为您的参数值
                api_name="/chat"
            )
            msgGPT = response
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
            print(msg)
            print(msgGPT)

            data = {"diff_id": diff_id, "msg": f"{msg}", "msgGPT": f"{msgGPT}"}
            # 获取 "msgGPT" 的内容
            msg_content = data.get("msgGPT", "")
            # 查找第一个 \" 的位置
            first_escape_start = msg_content.find('\"')
            first_escape_end = first_escape_start + 1
            # 查找第二个 \" 的位置
            second_escape_start = msg_content.find('\"', first_escape_end + 1)
            second_escape_end = second_escape_start + 1
            # 判断找到的位置是否合理
            if first_escape_start != -1 and first_escape_end != -1 and second_escape_start != -1 and second_escape_end != -1:
                # 更新 "msgGPT" 的内容
                updated_msgGPT = msg_content[first_escape_end:second_escape_start].strip()
                data["msgGPT"] = updated_msgGPT

            msg = data.get("msg")
            msgGPT = data.get("msgGPT")
            # 将 diff 和 msg ,score添加到列表中
            bleu_score = calculate_bleu(msg, msgGPT)
            rouge_l_score = calculate_rouge_l(msg, msgGPT)
            meteor_score = calculate_meteor(msg, msgGPT)
            print("METEOR Score:", meteor_score)
            print("BLEU Score:", bleu_score)
            print("ROUGE-L Score:", rouge_l_score)
            print("")

            # 将 diff 和 msg ,score添加到列表中
            data1 = {"METEOR Score": f"{meteor_score}",
                    "BLEU Score": f"{bleu_score}", "ROUGE-L Score": f"{rouge_l_score['f']}"}

            merged_dict = {**data, **data1}


            with open('java_final_no_result_7B7.jsonl', 'a',encoding='UTF-8') as f:
                json.dump(merged_dict, f)
                f.write('\n')
            time.sleep(2)
            break

        except:
            traceback.print_exc()
            merged_dict = {"diff_id": diff_id, "msg": f"0", "msgGPT": f"0"}
            with open('java_final_no_result_7B7.jsonl', 'a',encoding='UTF-8') as f:
                json.dump(data, f)
                f.write('\n')
            #time.sleep(5)
            attempts += 1
            if attempts == 3:
                print(f"{item} 已经重试了3次，仍然失败。")
                # 这里可以选择记录失败的item，或者是进行其他错误处理
                # ...
                break  # 重试达到3次后，跳出内部循环，处理下一个item




