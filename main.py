import gensim
from gensim.test.utils import common_texts
import jieba
from jieba.analyse import extract_tags
import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from urllib.request import urlretrieve
import re

def is_2letters(word):
    return True if re.match(r"^[a-zA-Z]{1,2}$", word) else False

def is_number(word):
    return True if re.match(r"^[0-9]+$", word) else False

def remove_stop_word(words):
    stop_words = set()
    with open("stop_words.txt", 'r', encoding='utf-8') as f:
        for w in f.readlines():
            stop_words.add(w.replace("\n", ""))
    new_words = [word for word in words if word not in stop_words and len(word) > 1 and not is_number(word) and not is_2letters(word) and word != "\r\n"]
    return new_words

def add_big_dic():
    # 使用大型字典
    big_dict_path = "dict.txt.big"
    if not os.path.exists(big_dict_path):
        print("下載大型字典")
        url = "https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big"
        urlretrieve(url, big_dict_path)
    jieba.set_dictionary(big_dict_path)

def alibaba(text):
    chars = [char for char in text if char ]
    return chars

def cut_head_and_tail(words, head_percentage, art_window):
    start_index = int(len(words)*head_percentage)
    end_index = start_index + art_window
    result = words[start_index: end_index]
    return result


def read_file(file):
    text = None
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def train_word2vec():
    add_big_dic()
    text = ''
    with open("article_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()
    model = Word2Vec(size=512, window=15, min_count=5, alpha=0.025, workers=6, iter=50, seed=0)
    model.save('w2v_1.model')
    print("Done")
    return model


def train():
    add_big_dic()
    df_train = pd.read_csv("products_202009101144.csv")
    text_list = list(df_train['description'])
    words_list = [remove_stop_word(jieba.lcut(str(text))) for text in text_list]
    documents = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(words_list)]
    model = Doc2Vec(documents, dm=1, dm_mean=1, vector_size=512,
                    window=15, min_count=5, alpha=0.025,
                    workers=6, epochs=50, dbow_words=1, seed=0)
    model.save('product5.model')
    print("Done")
    data = load_abc()
    return data

def load_abc():
    result = dict()
    products = list(pd.read_csv("products_202009101144.csv")['description'])
    articles = list(pd.read_csv("articles_202009101121.csv")['compact_content'])
    model = Doc2Vec.load('product5.model')
    model.random.seed(0)
    result['articles'] = articles
    result['products'] = products
    result['model'] = model
    return result

def suggest_gift(n, data, topn, start_percentage, art_window):
    products = data['products']
    if type(n) == type(1):
        article = data['articles'][n]
    else:
        article = read_file(n)
    model = data['model']
    result = dict()
    result['article_words'] = cut_head_and_tail(remove_stop_word(jieba.lcut(article)), start_percentage, art_window)
    result['article_vec'] = model.infer_vector(result['article_words'])
    sim = model.docvecs.most_similar([result['article_vec']], topn=topn)
    print([idx for (idx, rank) in sim])
    result['product_idx_list'] = [int(idx) for (idx, rank) in sim]
    result['products'] = [products[int(idx)] for (idx, rank) in sim]
    result['cur_article'] = article
    return result

def expain_why(source_text, target_text , max_depth, wv, topn, start_percentage, art_window):
    clean_words_short = cut_head_and_tail(remove_stop_word(jieba.lcut(source_text)), start_percentage, art_window)
    source_word_set = set(model_view(wv, " ".join(clean_words_short)))
    target_word_set = set(model_view(wv, target_text))
    print("source:", source_text, "\n", source_word_set)
    print("target:", target_text, "\n", target_word_set)
    reason = []
    for s in source_word_set:
        paths = search_paths(s, target_word_set, max_depth, wv, topn)
        reason += paths
    return reason

def search_paths(source_word, target_word_set, limit_depth, wv, topn):
    paths = []
    temp_paths = [[source_word]]
    i = 0
    while True:
        cur_path = temp_paths[i]
        if len(cur_path) == limit_depth:
            break
        next_words = get_similar_words(cur_path[-1], wv, topn)
        for n in next_words:
            new_path = cur_path + [n]
            temp_paths.append(new_path)
            if n in target_word_set:
                paths.append(new_path)
                print(source_word, "found:", new_path)
        i += 1
    return paths

def get_similar_words(word, wv_model, topn):
    words =[ w for w, _ in wv_model.most_similar([word], topn=topn)]
    return words

def model_view(model, text):
    after_stop_text_words = remove_stop_word(jieba.lcut(text))
    model_view_words = [ w for w in after_stop_text_words if w in model.wv.vocab]
    return model_view_words

def get_model_view_set(model, text):
    return set(model_view(model, text))

def create_corpus(articles):
    f = open("article_corpus.txt", "w", encoding="utf-8")
    for art in articles:
        a = " ".join(jieba.lcut(art))
        f.write(a)
    f.close()

def main():
    # train()
    # predict()
    suggest_gift()
if __name__ == '__main__':
    main()

