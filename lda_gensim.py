import random
import pandas as pd
import numpy as np
import jieba
import gensim
import matplotlib.pyplot as plt



def read_cn_stopwords():
    cn_stopwords = []
    with open('./cn_stopwords.txt', mode='r', encoding='utf-8') as f:
        cn_stopwords.extend([line.strip() for line in f.readlines()])
    return cn_stopwords


def read_novel(path): 
    para_list = []
    para_label = []
    # 读取小说列表
    with open(file=path + 'inf.txt', mode='r', encoding='gb18030') as f:
        file_names = f.readline().split(',')
    f.close()
    # 读取小说
    for index, name in enumerate(file_names):
        novel_name = path + name +'.txt'
        with open(file=novel_name, mode='r', encoding='gb18030') as f:
            content = f.read()
            ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
            content = content.replace(ad, '')
            for sentence in content.split('\n'):
                if len(sentence) < 500:
                    continue
                para_list.append(sentence)
                para_label.append(index)

    return para_list, para_label

def para_extract(para,label):
    # 段落抽取
    text_ls = []
    text_label = []
    random_indices = random.sample(range(len(para)), 200)
    text_ls.extend([para[i] for i in random_indices])
    text_label.extend([label[i] for i in random_indices])
    return text_ls,text_label

def split(text_ls,text_label):
    # 分词
    stop_words = read_cn_stopwords()
    tokens_word = []  # 以词为单位
    tokens_word_label = []
    tokens_char = []  # 以字为单位
    tokens_char_label = []
    for i, text in enumerate(text_ls):
        words = [word for word in jieba.lcut(sentence=text) if word not in stop_words]
        tokens_word.append(words)
        tokens_word_label.append(text_label[i])

        temp = []
        for word in words:
            temp.extend([char for char in word])
        tokens_char.append(temp)
        tokens_char_label.append(text_label[i])

    #词典与向量集
    dic_word = gensim.corpora.Dictionary(tokens_word)
    dic_char = gensim.corpora.Dictionary(tokens_char)

    corp_word = [dic_word.doc2bow(tokens) for tokens in tokens_word]
    corp_char = [dic_char.doc2bow(tokens) for tokens in tokens_char]
    return dic_word,dic_char,corp_word,corp_char,tokens_word,tokens_char

class LDA_model:
    def __init__(self,num_topics_list,text_ls,text_label):
        self.text_ls=text_ls
        self.text_label=text_label
        self.num_topics_list=num_topics_list
        self.pp_word_list = []
        self.cv_word_list = []
        self.pp_char_list = []
        self.cv_char_list = []

    def train(self,num_topics_list,dic_word,dic_char,corp_word,corp_char,tokens_word,tokens_char):
        for num_topics in num_topics_list:
            # 基于词
            lda_word = gensim.models.ldamodel.LdaModel(corpus=corp_word, id2word=dic_word, num_topics=num_topics,
                                                      passes=20, alpha='auto', eta='auto')

            perplexity_word = -lda_word.log_perplexity(corp_word)
            cv_model_word = gensim.models.CoherenceModel(model=lda_word, texts=tokens_word, dictionary=dic_word,
                                                        coherence='c_v')   
            self.pp_word_list.append(perplexity_word)
            self.cv_word_list.append(cv_model_word.get_coherence())
            # 基于字
            lda_char = gensim.models.ldamodel.LdaModel(corpus=corp_char, id2word=dic_char, num_topics=num_topics,
                                                      passes=20, alpha='auto', eta='auto')
            perplexity_char = -lda_char.log_perplexity(corp_char)
            cv_model_char = gensim.models.CoherenceModel(model=lda_char, texts=tokens_char, dictionary=dic_char,
                                                        coherence='c_v')  
            self.pp_char_list.append(perplexity_char)
            self.cv_char_list.append(cv_model_char.get_coherence())
            print("Finish! 当前topic数量为:",num_topics)
        
    def result_plot(self,num_topics_list):
        bar_width=1
        num_topics_list=list(num_topics_list)
        a1 = [i-bar_width/2 for i in num_topics_list]
        a2 = [i+bar_width/2 for i in num_topics_list]
        print(num_topics_list)


        plt.bar(a1, self.pp_word_list, label='word',width=bar_width)
        plt.bar(a2, self.pp_char_list, label='char',width=bar_width)
        plt.title('perplexity')
        plt.xlabel("number of topics")
        plt.ylabel("perplexity")
        plt.legend()
        plt.savefig('perplexity.jpg')  #保存成jpg格式
        plt.show()
        
        plt.bar(a1, self.cv_word_list, label='word',width=bar_width)
        plt.bar(a2, self.cv_char_list, label='char',width=bar_width)
        plt.title('coherence')
        plt.xlabel("number of topics")
        plt.ylabel("coherence")
        plt.legend()
        plt.savefig('coherence.jpg')  #保存成jpg格式
        plt.show()
if __name__ == '__main__':
    path = './jyxstxtqj_downcc.com/'
    para, label = read_novel(path)


    text_ls,text_label=para_extract(para,label)
    dic_word,dic_char,corp_word,corp_char,tokens_word,tokens_char=split(text_ls,text_label)
    num = range(2, 51, 8)
    LDA = LDA_model(num,text_ls,text_label)
    
    LDA.train(num,dic_word,dic_char,corp_word,corp_char,tokens_word,tokens_char)
    LDA.result_plot(num)
