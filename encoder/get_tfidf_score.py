import json
import re
import os
from copy import deepcopy
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt')


class TF_IDF_Model(object):
    def __init__(self, documents_list):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.tf = []
        self.idf = {}
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log(self.documents_number / (value + 1))

    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]
        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list


def remove(text):
    remove_chars = r"[^\w ]+|( *How *)|( *how *)"
    return re.sub(remove_chars, '', text)
        
def string2time(timestamp):
    return (int(timestamp[3])*60+int(timestamp[5:7]), int(timestamp[11])*60+int(timestamp[13:15]))

def get_sentences(sample, data_path):
    script_path = os.path.join(data_path, sample, "script.txt")
    with open(script_path) as f:
        sentences = f.readlines()
    sentences = [sentence[16:].strip().split('. ') for sentence in sentences if len(sentence)>15]
    sentences = sum(sentences, [])
    return sentences


def get_paras(sample, data_path):
    # define split pattern
    split_pattern = ('To', 'to', 'First', 'If', 'IN this case', 'In this case', 'By', 'Else,', 'How to', 'You can', 'in order to', 'This is how to')
    
    script_path = os.path.join(data_path, sample, 'script.txt')
    with open(script_path, 'r') as f:
        sentences = f.readlines()
    sentences_text = [sentence[16:].strip() for sentence in sentences if len(sentence)>15]
    sentecens_time = [sentence[:16].strip() for sentence in sentences if len(sentence)>15]

    # sentence to paragraph
    paras = []
    timestamps = []
    for i, (sentence, time) in enumerate(zip(sentences_text, sentecens_time)):
        if i == 0:
            para = []
            timestamp = []
        if sentence.startswith(split_pattern):
            paras.append(para)
            timestamps.append(timestamp)
            para = []
            timestamp = []
        para.append(sentence)
        timestamp.append(time)
        if i == len(sentences_text)-1:
            paras.append(para)
            timestamps.append(timestamp)
    paras = [i for i in paras if len(i)!=0]
    paras = [''.join(para) for para in paras]
    timestamps = [i for i in timestamps if len(i)!=0]
    timestamps = [[string2time(i) for i in timestamp] for timestamp in timestamps]
    timestamps = [(i[0][0], i[-1][1]) for i in timestamps]

    assert len(paras) == len(timestamps)
    return paras, timestamps

def question_to_sentence(sentences, qa_pair):
    sentences = [word_tokenize(remove(sentence)) for sentence in sentences]
    model = TF_IDF_Model(sentences)
    question = word_tokenize(remove(qa_pair['question']))
    scores = model.get_documents_score(question)
    return scores

def question_to_para(paras, qa_pair):
    paras = [word_tokenize(remove(para)) for para in paras]
    model = TF_IDF_Model(paras)
    question = word_tokenize(remove(qa_pair['question']))
    scores = model.get_documents_score(question)
    return scores


if __name__ == "__main__":    
    paths = ['/home/hy/ssd1/tomchen/loveu2023/dataset/assistq_test2023', '/home/hy/ssd1/tomchen/loveu2023/dataset/assistq_train']
    names = ['test2023_without_gt', 'train']

    for path, name in zip(paths, names):
        with open(os.path.join(path, name+'.json'), 'r') as f:
            ann = json.load(f)
        ann_with_score = deepcopy(ann)
        data_path = os.path.join(path, path.split('_')[-1])
        
        for sample in ann:
            sentences = get_sentences(sample, data_path)
            paras, paras_timestamps = get_paras(sample, data_path)
            with open(os.path.join(data_path, sample, 'paras.json'), 'w') as f:
                json.dump([paras_timestamps, paras], f)
            qa_pairs = ann[sample]
            qa_pairs_refine = ann_with_score[sample]
            for qa_pair, qa_pair_refine in zip(qa_pairs, qa_pairs_refine):
                paras_score = question_to_para(paras, qa_pair)
                sents_score = question_to_sentence(sentences, qa_pair)
                assert len(sentences) == len(sents_score)
                assert len(paras) == len(paras_score)
                qa_pair_refine['sents_score'] = sents_score
                qa_pair_refine['paras_score'] = paras_score

        with open(os.path.join(path, name+'_with_score.json'), 'w') as f:
            json.dump(ann_with_score, f)
