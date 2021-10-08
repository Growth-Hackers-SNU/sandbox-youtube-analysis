import re
import itertools
from kiwipiepy import Kiwi
import pandas as pd
from soynlp.hangle import compose, decompose, character_is_korean
from tqdm.notebook import tqdm

def generate_D(df, num):
    D_list = []
    for i in range(num):
        my_list = df[df['class'] == i]['tokenized'].to_list()
        D_list.append(list(itertools.chain.from_iterable(my_list)))

    return D_list

def jamo_sentence(sent):
    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        if len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else '_' for c in cjj)
        return cjj_

    sent_ = []
    for char in sent:
        if character_is_korean(char):
            sent_.append(transform(char))
        else:
            sent_.append(char)
    doublespace_pattern = re.compile('\s+')
    sent_ = doublespace_pattern.sub(' ', ''.join(sent_))
    return sent_


def eumjul_sentence(sent):
    tokenized = []
    for j in range(len(sent)):
        tokenized.append(sent[j])
    return tokenized


def jamo_to_word(jamo):
    jamo_list, idx = [], 0
    while idx < len(jamo):
        if not character_is_korean(jamo[idx]):
            jamo_list.append(jamo[idx])
            idx += 1
        else:
            jamo_list.append(jamo[idx:idx + 3])
            idx += 3

    word = ""
    for jamo_char in jamo_list:
        if len(jamo_char) == 1:
            word += jamo_char
        elif jamo_char[2] == "_":
            word += compose(jamo_char[0], jamo_char[1], " ")
        else:
            word += compose(jamo_char[0], jamo_char[1], jamo_char[2])

    return word


def transform(list_):
    to_return = []
    for w, r in list_:
        try:
            to_return.append((jamo_to_word(w), r))
        except:
            to_return.append((w, r))
    return to_return

def token_ready(Data, mode='kiwi', out_pos=['JKO', 'JKG', 'JKS', 'JKB', 'VCP', 'EC', 'EF', 'JX'], min_sup=1, min_word=1):
    ### 토큰화 준비, 필수로 실행되어야 하며, df_ready 후 실행 가능
    # token : kiwi / fasttext / fasttext_eumjul 중 선택(fasttext는 자모 단위)
    # out_pos : kiwi에서만 유효, 제거 품사 선택
    # min_sup, min_word : kiwi에서만 유효, 각각 데이터 프레임 내 최소 등장 횟수, 최소 글자 수 의미

    try:
        if mode in ['kiwi', 'fasttext', 'fasttext_eumjul']:
            pass
        else:
            raise Exception
    except:
        print("Invalid mode! valid token list : ['kiwi', 'fasttext', 'fasttext_eumjul']")

    try:
        comment_list = list(Data.comment)
        tmp_list = []  # 문장부호 삭제
        kiwi_list = []
        result = Data.copy()
    except:
        print('Wrong Data')


    if mode == 'kiwi':

        print('Processing 1/2...')

        kiwi = Kiwi()
        kiwi.prepare()

        for i in tqdm(range(len(comment_list))):
            try:
                tmp_list.append([kiwi.analyze(
                    re.sub(r'["!#$%&\'()0-9*+,-./:;<=>?@\[\]^_\`{|}~\\\\]', '', comment_list[i]))])  # 문장 부호 삭제
            except:
                tmp_list.append([])

        print('Processing 2/2...')

        for i in range(len(tmp_list)):
            temp_word = []
            if len(tmp_list[i]) > 0:
                for j in range(len(tmp_list[i][0][0][0])):
                    if tmp_list[i][0][0][0][j][1] not in out_pos and len(
                            tmp_list[i][0][0][0][j][0]) >= min_word:  # min_word보다 짧은 글자 수의 단어 제거
                        temp_word.append(tmp_list[i][0][0][0][j][0])
            kiwi_list.append(temp_word)

        for i in range(len(result)):
            result['tokenized'] = kiwi_list

    elif mode == 'fasttext':
        # Fasttext 자모 단위 모델
        result['tokenized'] = Data.apply(lambda row: jamo_sentence(str(row['comment'])).strip().split(" "), axis=1)

    elif mode == 'fasttext_eumjul':
        # Fasttext 음절 단위 모델
        result['tokenized'] = Data.apply(lambda row: eumjul_sentence(str(row['comment'])), axis=1)

    D_list = generate_D(result, 7)

    words = []
    for i in range(len(D_list)):
        words.append(list(set(D_list[i])))

    words = list(itertools.chain.from_iterable(words))

    if min_sup > 1:  # 단어의 최소 등장 횟수(전체 댓글 내에서, 7개 감정 동일 샘플링 데이터프레임 기준)
        count_dict = {}
        for word in words:
            try:
                count_dict[word] += 1
            except:
                count_dict[word] = 1
        count_ser = pd.Series(count_dict)
        count_ser = count_ser[count_ser >= min_sup]
        words = list(count_ser.index)

        for i in range(len(result)):
            token_list = result.loc[i, 'tokenized']
            valid = []
            for t in token_list:
                if t in words:
                    valid.append(t)
            result.loc[i, 'tokenized'] = valid

    print("Tokenization Done!")
    return result