import json
import os
import re
import time
import traceback
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize

# 初始化NLTK的组件
nltk.download('punkt')

def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s

def to_Underline(x):
    """转空格命名"""
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', ' \g<0>', x).lower()

def get_tokens(text):
    tokens = word_tokenize(text)
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

# 设置Gemini API
genai.configure(api_key="AIzaSyDFhn3cI0UR4eo2cT90RAuJCzx59AwZGwQ")
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_output_tokens": 50
}
model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)

# 定义输出文件名
output_filenames = 'pyaddresult_gemini.jsonl'

# 读取并处理代码差异数据
with open('py1.jsonl', 'r') as f:
    json_data = f.readlines()

for item in json_data:
    data = json.loads(item)
    diff_id = data['diff_id']
    diff = data['diff']
    # 应用预处理步骤
    result = remove_between_identifiers(diff, 'mmm a', '<nl>')
    diff = get_tokens(remove_between_identifiers(result, 'ppp b', '<nl>'))

    if len(best_diffs_msgs) >= num_examples:
        # 构建prompt
        prompt = ""
        for best_diff, best_msg in best_diffs_msgs[:num_examples]:
            prompt += f"{best_diff}\nPlease write a commit message for the above code change.\n{best_msg}\n\n"
        prompt += f"{diff}\nPlease write a commit message for the above code change.\n"

        generated_msg = None
        attempt = 0
        while attempt < 10 and generated_msg is None:
            try:
                response = model.generate_content([prompt])
                generated_msg = response.text.strip()
            except Exception as e:
                print(f"尝试 {attempt + 1} 失败: {e}")
                attempt += 1
                time.sleep(1)  # 简单的等待机制，以避免过快重试

        # 如果成功生成了消息或达到最大尝试次数，将结果保存到相应的文件中
        if generated_msg is not None:
            output_data = {
                "diff_id": diff_id,
                f"generated_msg_{num_examples}": generated_msg
            }
            with open(output_filenames[num_examples], 'a', encoding='utf8') as f:
                json.dump(output_data, f)
                f.write('\n')
        else:
            print(f"无法为diff_id {diff_id}生成消息。")
    time.sleep(1)