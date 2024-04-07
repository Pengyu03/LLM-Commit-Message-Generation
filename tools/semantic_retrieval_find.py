import json
import numpy as np
from numpy.linalg import norm
from collections import defaultdict
import time


vlan = 'vpy1no.jsonl'
vtrain = 'encoded_diffspy2.jsonl'
output_file = 'pybest_no_selectv.jsonl'
input_file = 'pytrain_no_selectv.jsonl'

def load_vectors(jsonl_file):
    """从JSONL文件中加载diff_id和向量"""
    vectors = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            vectors[data['diff_id']] = np.array(data['cls_vector'])
    return vectors

def cosine_similarity(vec_a, vec_b):
    """计算两个向量的余弦相似度"""
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

def get_top_similar(test_vec, train_vectors, top_k=10):
    """获取与测试向量最相似的top_k个训练向量的diff_id"""
    similarities = {train_id: cosine_similarity(test_vec, train_vec)
                    for train_id, train_vec in train_vectors.items()}
    # 根据相似度排序并取前top_k个
    return sorted(similarities, key=similarities.get, reverse=True)[:top_k]

def get_diff_msg(jsonl_file, diff_ids):
    """从JSONL文件中获取特定diff_id的diff和msg"""
    diff_msg = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['diff_id'] in diff_ids:
                diff_msg[data['diff_id']] = {'diff': data['diff'], 'msg': data['msg']}
    return diff_msg


# 假设load_vectors和get_top_similar已经定义

# 加载测试集和训练集向量
test_vectors = load_vectors(vlan)
train_vectors = load_vectors(vtrain)

with open(output_file, 'a') as outfile:
    for test_id, test_vec in test_vectors.items():

        # 获取最相似的top 10个训练集diff_id
        top_similar_ids = get_top_similar(test_vec, train_vectors, top_k=10)

        # 从pytrainyuan3.jsonl获取这些diff_id的diff和msg
        diff_msg_data = get_diff_msg(input_file, top_similar_ids)

        # 格式化结果
        result = {"diff_id": test_id}
        for i, similar_id in enumerate(top_similar_ids, 1):
            result[f"best_id{i}"] = similar_id
            result[f"best_diff{i}"] = diff_msg_data[similar_id]['diff']
            result[f"best_msg{i}"] = diff_msg_data[similar_id]['msg']

        # 写入单条结果到文件
        outfile.write(json.dumps(result) + '\n')
