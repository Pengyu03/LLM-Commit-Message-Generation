import json
import jsonlines
import re
import nltk

# 读取输入 JSONL 文件
input_file = 'input.jsonl'
output_file = 'output.jsonl'
other_output_file = 'other_data.jsonl'
lan = ''

with open(input_file, 'r') as infile, open(lan_output_file, 'w') as lan_outfile, open(other_output_file, 'w') as other_outfile:
    for line in infile:
        data = json.loads(line)
        if 'diff' in data:
            diff_text = data['diff']
            start_index = diff_text.find("ppp b ")
            end_index = diff_text.find("<nl>", start_index)
            if start_index != -1 and end_index != -1:
                extracted_data = diff_text[start_index + len("ppp b "):end_index]
                dot_index = extracted_data.rfind(".")
                if dot_index != -1:
                    content_between_dot_nl = extracted_data[dot_index + 1:]
                    if content_between_dot_nl == " java ":
                        lan_outfile.write(json.dumps(data) + '\n')
                    elif content_between_dot_nl == " cs ":
                        lan_outfile.write(json.dumps(data) + '\n')
                    elif content_between_dot_nl == " js ":
                        lan_outfile.write(json.dumps(data) + '\n')
                    elif content_between_dot_nl == " py ":
                        lan_outfile.write(json.dumps(data) + '\n')
                    elif content_between_dot_nl == " cpp ":
                        lan_outfile.write(json.dumps(data) + '\n')
                    else:
                        other_outfile.write(json.dumps(data) + '\n')

input_file1 = lan_output_file

with open(input_file1, 'r') as infile, open(output_file1, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        if 'diff' in data:
            diff_text = data['diff']
            start_index = diff_text.find("ppp b ")
            end_index = diff_text.find("<nl>", start_index)
            if start_index != -1 and end_index != -1:
                extracted_data = diff_text[start_index + len("ppp b "):end_index]
                # 仅保留最后一个 "/" 和 "." 之间的内容
                last_slash_index = extracted_data.rfind("/")
                last_dot_index = extracted_data.rfind(".")
                if last_slash_index != -1 and last_dot_index != -1:
                    extracted_data = extracted_data[last_slash_index + 1:last_dot_index]
                # 将提取的数据添加到原数据集中，覆盖 "file_name"
                data['file_name'] = extracted_data
        # 写入包含处理后的数据的原数据集到输出文件
        outfile.write(json.dumps(data) + '\n')

# 读取输入 JSONL 文件
input_file2 = output_file1
output_file2 = '2.jsonl'

with open(input_file2, 'r') as infile, open(output_file2, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        if 'msg' in data and 'file_name' in data:
            msg = data['msg']
            file_name = data['file_name']
            # 检查"msg"中是否包含"file_name"对应的内容
            if file_name in msg:
                # 用"<file_name>"替换msg中原对应内容的字符数据
                msg = msg.replace(file_name, ' <file_name> ')
            # 更新数据中的"msg"字段
            data['msg'] = msg
        # 将更新后的数据写入输出文件
        outfile.write(json.dumps(data) + '\n')

input_file3 = output_file2
output_file3 = '3.jsonl'
# 定义一个函数，接受diff作为参数，返回一个包含函数名的列表
def extract_function_names(diff):
    # 定义一个空列表，用于存储函数名
    global lan
    function_names = []
    # 定义一个正则表达式，用于匹配返回值类型和函数名
    if lan == 'java'or'cpp'or'csharp':
        pattern = r"\w+\s+(\w+)\s*\("
    if lan == 'py':
        pattern = r'def\s+(\w+)'
    if lan == 'js':
        pattern = r'function\s+(\w+)'
    # 用re模块的findall方法，在diff中查找所有符合pattern的字符串，得到一个列表
    matches = re.findall(pattern, diff)
    # 遍历matches列表中的每个字符串
    for match in matches:
        # 把字符串添加到function_names列表中
        function_names.append(match)
    # 返回function_names列表
    return function_names

# 定义一个函数，接受输入文件名和输出文件名作为参数，进行处理和保存
def process_jsonl(input_file, output_file):
    # 用jsonlines模块打开输入文件，得到一个reader对象
    with jsonlines.open(input_file) as reader:
        # 用jsonlines模块打开输出文件，得到一个writer对象
        with jsonlines.open(output_file, mode='w') as writer:
            # 遍历reader对象中的每一条json
            for obj in reader:
                # 取出diff属性的值
                diff = obj['diff']
                # 调用extract_function_names函数，得到一个包含函数名的列表
                function_names = extract_function_names(diff)
                # 把obj中添加一个function_names属性，值为function_names列表
                obj['function_names'] = function_names
                #print(function_names)
                # 用writer对象把obj写入输出文件中
                writer.write(obj)
# 测试函数
process_jsonl(input_file3, output_file3)

input_file4 = output_file3
output_file4 = '4.jsonl'

def replace_function_names(msg, function_names):
    for function_name in function_names:
        if function_name in msg:
            msg = msg.replace(function_name, '<method_name>')
    return msg

with open(input_file4, 'r',encoding='UTF-8') as infile, open(output_file4, 'w',encoding='UTF-8') as outfile:
    for line in infile:
        data = json.loads(line)
        if 'msg' in data and 'function_names' in data:
            msg = data['msg']
            function_names = data['function_names']
            # 替换msg中包含在function_names列表中的内容
            msg = replace_function_names(msg, function_names)
            # 更新数据中的"msg"字段
            data['msg'] = msg
        # 将更新后的数据写入输出文件
        outfile.write(json.dumps(data) + '\n')

# 定义一个函数，接受msg和diff作为参数，返回替换后的msgnew
def replace_token(msg, diff):
    # 用nltk的word_tokenize方法对msg和diff进行分词，得到两个列表
    msg_tokens = nltk.word_tokenize(msg)
    diff_tokens = nltk.word_tokenize(diff)
    # 定义一个空列表，用于存储替换后的msg的token
    msgnew_tokens = []
    # 遍历msg的token
    for token in msg_tokens:
        # 如果这个token在diff的token中出现过，就把它替换为<iden>
        if (token in diff_tokens) and len(token) > 5 and (token != '<file_name>') and (token != '<method_name>') :

            token = "<iden>"
        # 把替换后的token添加到列表中
        msgnew_tokens.append(token)
    # 用空格把列表中的token连接起来，得到msgnew
    msgnew = " ".join(msgnew_tokens)
    # 返回msgnew
    return msgnew

input_file5 = output_file4
output_file5 = '5.jsonl'
# 定义一个函数，接受输入文件名和输出文件名作为参数，进行处理和保存
def process_jsonl(input_file, output_file):
    # 用jsonlines模块打开输入文件，得到一个reader对象
    with jsonlines.open(input_file) as reader:
        # 用jsonlines模块打开输出文件，得到一个writer对象
        with jsonlines.open(output_file, mode='w') as writer:
            # 遍历reader对象中的每一条json
            for obj in reader:
                # 取出msg和diff属性的值
                msg = obj['msg']
                diff = obj['diff']
                # 调用replace_token函数，得到msgnew
                msgnew = replace_token(msg, diff)
                #print(msgnew)
                # 把obj中的msg属性替换为msgnew属性
                obj.pop('msg')
                obj['msg'] = msgnew

                # 用writer对象把obj写入输出文件中
                writer.write(obj)
# 测试函数
process_jsonl(input_file5, output_file5)

input_file6 = output_file5
output_file6 = '6.jsonl'

def replace_method_name(msg):
    # 替换"< method_name >"为"<method_name>"
    msg = msg.replace('< method_name >', '<method_name>')
    msg = msg.replace('< file_name >', '<file_name>')
    return msg

with open(input_file6, 'r',encoding='UTF-8') as infile, open(output_file6, 'w',encoding='UTF-8') as outfile:
    for line in infile:
        data = json.loads(line)
        if 'msg' in data:
            msg = data['msg']
            # 替换"< method_name >"为"<method_name>"
            msg = replace_method_name(msg)
            # 更新数据中的"msg"字段
            data['msg'] = msg
        # 将更新后的数据写入输出文件
        outfile.write(json.dumps(data) + '\n')

jsonl_file_path = output_file6
output_jsonl_file_path = '6.jsonl'

def select_message_from_jsonl():
    # 从JSONL文件中加载数据并返回
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    samples = []
    for line in lines:
        data = json.loads(line)
        #id = data['diff_id']
        message = data['msg']
#        file_names = data['file_names']
        samples.append(message)
    return samples

def update_jsonl_file(samples):
    # 将处理后的数据写回JSONL文件
    with open(output_jsonl_file_path, 'w', encoding='utf-8') as file:
        for sample in samples:
            data = {
                #'diff_id': sample[0],
                'msg': sample,
#                'file_names': sample[2]
            }
            file.write(json.dumps(data, ensure_ascii=False) + '\n')

def find_url(message):
    if 'git-svn-id: ' in message:
        # For git-svn-id links, handle them separately
        pattern = re.compile(r'git-svn-id:\s+(?:http[s]?\s:\s/\s/\s(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\s+(?:[a-z]|[0-9])+(?:-(?:[a-z]|[0-9])+){4}\s+)')

    else:
        pattern = re.compile(r'(http[s]?\s:\s/\s/\s(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\s+)')
    urls = re.findall(pattern, message)
    urls = sorted(list(set(urls)), reverse=True)
    for url in urls:
        message = message.replace(url, '<link>')
    return message

def find_version(message):
    pattern = re.compile(r'[vVr]?\d+(?:\.\w+)+(?:-(?:\w)*){1,2}')
    versions = pattern.findall(message)
    versions = sorted(list(set(versions)), reverse=True)
    for version in versions:
        message = message.replace(version, '<version>')

    pattern2 = re.compile(r'[vVr]?\d+(?:\s\.\s\w+)+')
    versions = pattern2.findall(message)
    # Remove duplicate pattern
    versions = sorted(list(set(versions)), reverse=True)
    for version in versions:
        message = message.replace(version, '<version>')
    return message

def find_enter(message):
    pattern = re.compile(r'<nl>')
    enters = pattern.findall(message)
    enters = sorted(list(set(enters)), reverse=True)
    for enter in enters:
        message = message.replace(enter, '<enter>')
    return message

def find_table(message):
    pattern = re.compile(r'\t')
    tables = pattern.findall(message)
    tables = sorted(list(set(tables)), reverse=True)
    for table in tables:
        message = message.replace(table, '<tab>')
    return message

def find_rawCode(message):
    rawCodeSta = message.find('```')
    replaceIden = []
    res = ''
    while rawCodeSta > 0:
        rawCodeEnd = message.find('```', rawCodeSta + 3, len(message))
        if rawCodeEnd != -1:
            replaceIden.append([rawCodeSta, rawCodeEnd + 3])
        else:
            break
        rawCodeSta = message.find('```', rawCodeEnd + 3, len(message))
    if len(replaceIden) > 0:
        end = 0
        for iden in replaceIden:
            res += message[end:iden[0]]
            end = iden[1]
        res += message[end:len(message)]
        return res
    else:
        return message

if __name__ == '__main__':
    messages = []
    samples = select_message_from_jsonl()
    for sample in samples:
        #sample = sample.replace(' ','')
        message = find_url(sample)
        message = find_version(message)
        message = find_rawCode(message)
        message = find_enter(message)
        message = find_table(message)
        messages.append(message)
        #sample = str(sample)

    #sample ="src/org/junit/experimental/theories/Theories.java- Moved InitializationError to ParentRunner, since it was only used by <enter>   subclasses of ParentRunner. <enter> - Broke up TestMethod into FrameworkMethod (which makes it more clear <enter>   that these methods can also be Before, After, etc.), and <enter>   TestAnnotation (for specific information only available on the @Test <enter>   annotation). <enter> - Created TestMethodElement to encapsulate the relationship between <enter>   @Test, @Before, and @After.  This class may go away again quickly <enter> - Updated version in docs to 4.5 <enter> - Included docs about junit-dep jar"
    #sample = ['1','- Moved InitializationError to ParentRunnerhThis class may go away again quickly <enter>','src/org/junit/experimental/theories/Theories.java']
    #sample = sample.split(" ")
    #sample = sample.replace(' ','')
    #b = replace_file_name(sample)
    #print(b)

    update_jsonl_file(messages)

# 读取第一个和第二个JSONL文件
input_file_1 = output_file6
input_file_2 = output_jsonl_file_path

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data.append(json.loads(line))
    return data

data_1 = load_jsonl(input_file_1)
data_2 = load_jsonl(input_file_2)

# 替换第一个文件中的"msg"内容
if len(data_1) == len(data_2):
    for i in range(len(data_1)):
        data_1[i]['msg'] = data_2[i]['msg']

# 写入输出文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for item in data_1:
        outfile.write(json.dumps(item) + '\n')
