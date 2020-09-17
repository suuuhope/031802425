
import jieba
import jieba.analyse
from sklearn.metrics.pairwise import cosine_similarity
import sys


class CosineSimilarity(object):
    # 对CosineSimilarity的构造
    def __init__(self, file1, file2):
        self.s1 = file1
        self.s2 = file2

    @staticmethod   # 静态函数
    def get_key(content):  # 提取关键词
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']  # 分词
        # 按照权重返回前topK个关键词
        key = jieba.analyse.extract_tags("|".join(seg), topK=K, withWeight=False)
        return key

    @staticmethod   # oneHot编码
    def one_hot(key_dict, key):
        cut_code = [0] * len(key_dict)
        for word in key:
            cut_code[key_dict[word]] += 1
        return cut_code

    def calculate(self):
        # 提取关键词
        key1 = self.get_key(self.s1)
        key2 = self.get_key(self.s2)
        # 词的并集
        union = set(key1).union(set(key2))
        # 构造hash表
        key_dict = {}
        i = 0
        for word in union:
            key_dict[word] = i
            i += 1
        # oneHot编码
        vector1 = self.one_hot(key_dict, key1)
        vector2 = self.one_hot(key_dict, key2)
        sample = [vector1, vector2]
        # 除0处理
        try:
            simRate = cosine_similarity(sample)   # 用sklearn自带的余弦相似度计算
            return simRate[1][0]
        except Exception as e:
            print(e)
            return 0.0


# 测试
if __name__ == '__main__':
    # 命令行输入绝对路径
    root_Path = sys.argv[1]
    copy_Path = sys.argv[2]
    ans_Path = sys.argv[3]
    # 读入两个文本 计算topK
    try:
        with open(root_Path, encoding='UTF-8') as fp:
            root = fp.read()
            seg = [i for i in jieba.cut(root, cut_all=True) if i != '']
        K = int(len(seg) / 8)
    except:
        K = 0
    try:
        with open(root_Path, encoding='UTF-8') as fp:
            ori = fp.read()
        with open(copy_Path, encoding='UTF-8') as fp:
            copy = fp.read()
    except Exception as e:
        print(e)
    model = CosineSimilarity(ori, copy)
    # 保留两位小数
    similarity = round(model.calculate(), 2)
    # 输出答案到文本
    try:
        with open(ans_Path, "w+", encoding='UTF-8') as fp:
            fp.write(str(similarity))
    except Exception as e:
        print(e)