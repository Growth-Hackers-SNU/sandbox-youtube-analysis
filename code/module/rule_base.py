def generate_D(doc):
    D_list = []
    posneg = [1, 2]
    for i in posneg:
        my_list = doc[doc['pos_neg_neu'] == i]['token'].to_list()
        D_list.append(list(itertools.chain.from_iterable(my_list)))

    return D_list


def TwF(D, doc, t, label):
    d = doc[doc['pos_neg_neu'] == label].reset_index(drop=True)
    d_list = D[label - 1]
    n = 0

    for i in range(len(D)):
        if t in D[i]:
            n += 1
        else:
            continue

    data_dic = d['token'].to_dict()
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


def rule_dict(Data, class_num = 3):
    D_list = generate_D(Data)

    pos_dic = {}
    for k in tqdm(list(set(D_list[0]))):
        value = TwF(D_list, Data, k, 1)
        pos_dic[k] = value
    pos_dic_sorted=sorted(pos_dic.items(), key=(lambda x: x[1]), reverse=True)

    neg_dic = {}
    for k in tqdm(list(set(D_list[1]))):
        value = TwF(D_list, Data, k, 2)
        neg_dic[k] = value
    neg_dic_sorted=sorted(neg_dic.items(), key=(lambda x: x[1]), reverse=True)

    return pos_dic_sorted, neg_dic_sorted


def rule_dict_norm(Data, class_num=3):
    vocab = {}
    vocab_emotion = [{}, {}, {}, {}, {}, {}, {}]
    kiwi_list = list(Data['token'])

    for i, word_list in enumerate(kiwi_list):
        class_ = Data.loc[i, 'class'] - 1
        for word in word_list:
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1
            try:
                vocab_emotion[class_][word] += 1
            except:
                vocab_emotion[class_][word] = 1

    corp_df = pd.DataFrame.from_dict(vocab, orient='index', columns=['total_count']).sort_values('total_count',
                                                                                                 ascending=False)
    for i in range(1, 8):
        corp_df['class_%d_count' % i] = pd.Series(vocab_emotion[i - 1])

    corp_df = corp_df.fillna(0)
    for i in range(1, 8):
        corp_df['class_%d_ratio' % i] = np.round(100 * (corp_df['class_%d_count' % i] / corp_df['total_count']), 2)

    ratio = Data.groupby('class')['class'].count()
    for i in range(1, 8):
        corp_df['class_%d_prob' % i] = np.round(100 * (corp_df['class_%d_count' % i] / ratio[i]), 2)

    for i in range(1, 8):
        corp_df['class_%d_norm' % i] = np.round(100 * (corp_df['class_%d_prob' % i] / \
                                                       (corp_df['class_1_prob'] + corp_df['class_2_prob'] + corp_df[
                                                           'class_3_prob'] + corp_df['class_4_prob'] + corp_df[
                                                            'class_5_prob'] + corp_df['class_6_prob'] \
                                                        + corp_df['class_7_prob'])), 2)

    return corp_df[['total_count'] + list(corp_df.columns[22:])]


def rule_twf_test(Data, pos_dic_sorted, neg_dic_sorted):
    #  try:
    #    if mode is not in ['twf', 'norm_dict']:
    #      raise Exception
    #  except:
    #    print("wrong mode")

    pos_score = []
    neg_score = []
    for comment in Data['comment']:
        com_pos_score = 0
        com_neg_score = 0
        for i in range(len(pos_dic_sorted)):
            if pos_dic_sorted[i][0] in comment:
                com_pos_score += pos_dic_sorted[i][1]
        for j in range(len(neg_dic_sorted)):
            if neg_dic_sorted[j][0] in comment:
                com_neg_score += neg_dic_sorted[j][1]
        pos_score.append(com_pos_score)
        neg_score.append(com_neg_score)

    pos_neg_score = []
    for i in range(len(pos_score)):
        if pos_score[i] > neg_score[i]:
            pos_neg_score.append(1)
        elif pos_score[i] < neg_score[i]:
            pos_neg_score.append(2)
        else:
            pos_neg_score.append(0)

    Data['pos_neg_score'] = pos_neg_score


    return Data

def rule_norm_test(Data, norm_dict)
    norm_dict.rename(columns={'Unnamed: 0':'token'}, inplace=True)
    norm_dict['pos'] = norm_dict[['class_1_norm','class_2_norm','class_3_norm']].sum(axis=1)
    norm_dict['neg'] = norm_dict[['class_4_norm','class_5_norm','class_6_norm']].sum(axis=1)
    pos_neg_score = []
    token_list = list(Data['token'])

    for i in range(len(Data)):
        pos_score=0
        neg_score=0
        for word in token_list:
            if word in Data['comment'][i]:
                pos_score+=float(norm_dict[norm_dict['token']==word]['pos'])
                neg_score=float(neg_score+norm_dict[norm_dict['token']==word]['neg'])
            else:
                continue
        if pos_score>neg_score:
            pos_neg_score.append(1)
        elif pos_score<neg_score:
            pos_neg_score.append(2)
        else:
            pos_neg_score.append(0)
    Data['pos_neg_score'] = pos_neg_score
    return Data