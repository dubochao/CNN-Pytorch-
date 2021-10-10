import re
from torchtext.legacy import data
import jieba
import logging
jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[\u4e00-\u9fa5]|[a-zA-z]{3,8}|[0-9]{3,8}')
# stopword_file = './hit_stopwords.txt'

# 读取停用词列表
def get_stopword_list(file):
    with open(file, 'r', encoding='utf-8') as f:    #
        stopword_list = [word.strip('\n') for word in f.readlines()]
    return stopword_list
# 3. 导入停止词的语料库,
# 对文本进行停止词的去除
# stop = get_stopword_list(stopword_file)  # 获得停用词列表
def drop_stops(Jie_content, stop):
    # clean_content = []
    
    line_clean = []
    for line in Jie_content:
        if line in stop:
            continue
        line_clean.append(line)
    # clean_content.append(line_clean)
    return line_clean
def word_cut(text):
    line = ''.join(regex .findall(text))
    split_content = jieba.lcut(line)
    clean_content=[i for i in split_content if i != ' ']
    # clean_content = drop_stops(Jie_content, stop)
    return clean_content
def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train.tsv', validation='dev.tsv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    return train, dev
if __name__=='__main__':
    print(word_cut('操控性舒服、油耗低，性价比高'))

