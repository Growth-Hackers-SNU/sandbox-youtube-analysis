import itertools
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from gensim.models import FastText

def TwF(D, doc, t, label):
    d = doc[doc['class'] == label].reset_index(drop=True)
    d_list = D[label - 1]
    n = 0

    for i in range(len(D)):
        if t in D[i]:
            n += 1
        else:
            continue

    data_dic = d['tokenized'].to_dict()
    NTF = len([data_dic[q] for q in d.index.to_list() if t in data_dic[q]]) / len(list(set(d_list)))

    try:
        if (n < len(D)):
            weight = 1 / n
        else:
            weight = 0
    except:
        weight = 0

    twf = NTF * weight
    # print('NTF: {}/{}'.format(len([data_dic[q] for q in data.index.to_list() if t in data_dic[q]]), len(list(set(d_list)))))
    # print(n, NTF, weight)
    return twf

def generate_D(df, num):
    D_list = []
    for i in range(num):
        my_list = df[df['class'] == i]['tokenized'].to_list()
        D_list.append(list(itertools.chain.from_iterable(my_list)))

    return D_list


def vectorizer(word, doc, D, how):
    vector_list = []

    for i in tqdm(range(len(doc))):
        temp_vector = list(np.zeros(len(word)))
        token_list = doc.loc[i, 'tokenized']
        for t in token_list:
            try:
                index = word.index(t)
                if how == 'one_hot_cnt':
                    temp_vector[index] += 1
                elif how == 'one_hot':
                    temp_vector[index] = 1
                elif how == 'TwF':
                    temp_vector[index] = TwF(D, doc, t, doc.loc[i, 'class'])
            except:
                pass

        vector_list.append(temp_vector)

    return vector_list


def vectorizer_ftxt(comment, model):
    vector_list = []

    for i in tqdm(range(len(comment))):
        try:
            vector_list.append(model.wv.get_vector(tuple(comment.iloc[i])))
        except:
            vector_list.append([])

    return vector_list

def vectorize(Data, mode = 'one_hot', token = 'kiwi', directory = ""):
    try:
      if mode not in ['one_hot', 'one_hot_count', 'TwF', 'fasttext', 'fasttext_eumjul']:
        raise Exception
    except:
      print('Wrong mode!')

    try:
      if token not in ['kiwi', 'fasttext', 'fasttext_eumjul']:
        raise Exception
    except:
      print('Wrong Token!')


    D_list = generate_D(Data, 7)

    words = []
    for i in range(len(D_list)):
      words.append(list(set(D_list[i])))

    words = list(itertools.chain.from_iterable(words))

    if token == 'kiwi':
      vec = vectorizer(words, Data, D_list, mode)
    else:
      model = FastText.load(directory)
      vec = vectorizer_ftxt(Data.token, model)
    result = Data.copy()
    result['vector'] = vec
    print("Vectorization Done!")
    return result