from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import operator
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def evaluate(result_df, class_num=7):
    ### 모델, X_test, y_test를 input으로 넣으면 accuracy와 loss 값을 차례로 출력
    assert class_num in [3, 7], "Only 3 or 7 are allowed for class_num"

    pred = result_df['pred']
    acc = accuracy_score(pred, result_df['class'])

    y_test_label = list(result_df['class'])

    result_mat = []
    for i in range(class_num): result_mat.append([0] * class_num)
    for i in range(len(y_test_label)):
        result_mat[y_test_label[i]][pred[i]] += 1

    if class_num == 7:
        loss_mat = [[0, -1, -1, -1, -1, -1, -1],
                    [-1, 0, -1, -2, -4, -4, -2],
                    [-1, -1, 0, -2, -4, -4, -2],
                    [-1, -2, -2, 0, -2, -2, -2],
                    [-1, -4, -4, -2, 0, -1, -2],
                    [-1, -4, -4, -2, -1, 0, -2],
                    [-1, -2, -2, -2, -2, -2, 0]]
    else:
        loss_mat = [[0, -1, -2], [-1, 0, -1], [-2, -1, 0]]

    total_loss = 0.0
    for i in range(class_num):
        for j in range(class_num):
            total_loss += loss_mat[i][j] * result_mat[i][j]

    avg_loss = total_loss / np.sum(result_mat)

    return acc, avg_loss


def result_matrix(result_df, class_num=7):
    ### 각 라벨마다 실제 분류 결과에 대한 matrix를 보여주는 함수
    assert class_num in [3, 7], "Only 3 or 7 are allowed for class_num"

    pred = result_df['pred']
    y_test_label = list(result_df['class'])

    result_mat = []
    for i in range(class_num): result_mat.append([0] * class_num)
    for i in range(len(y_test_label)):
        result_mat[y_test_label[i]][pred[i]] += 1

    if class_num == 7:
        print(
            "real | predict    class 0      class 1      class 2      class 3      class 4      class 5      class 6      total")
        for i in range(class_num):
            print("class %d :" % (i), end="")
            for j in range(class_num):
                print("           %2d" % result_mat[i][j], end="")
            print("         %.2f%%" % (100 * result_mat[i][i] / sum(result_mat[i])))
    else:
        print("real | predict    Positive      Neutral      Negative      total")
        for i in range(class_num):
            print("%s :" % (['Positive', 'Neutral', 'Negative'][i]), end="")
            for j in range(class_num):
                print("           %2d" % result_mat[i][j], end="")
            print("         %.2f%%" % (100 * result_mat[i][i] / sum(result_mat[i])))


def result_df(mod, test_com, X_test, y_test, class_num=7):
    ### 세부 분류 데이터 프레임을 보여줌(댓글 - 실제 라벨 - 예측 라벨)
    # model에 학습이 이미 완료되었던 모델만 가능
    # voting model도 학습 시 지정했던 모델 이름을 가져와서 사용 가능
    # class_num = 3 과 7 중 선택
    # 결과 데이터 프레임 반환

    assert class_num in [3, 7], "Only 3 or 7 are allowed for class_num"

    pred = mod.predict(X_test)
    y_test = y_test.reset_index(drop=True)
    result = pd.DataFrame(test_com)
    result = result.reset_index(drop=True)

    result['class'] = y_test
    result['pred'] = pd.Series(pred)

    return result

def predict(mod, test_com, X_test, print_stat = True):
  # 새로운 데이터에 대해서 예측, 벡터화까지의 전처리 과정 필요
  # print_stat = True로 설정 시에 각 라벨별 예측 개수를 보여줌

  pred = mod.predict(X_test)
  test_com = pd.DataFrame(test_com)
  test_com = test_com.reset_index(drop = True)

  test_com['pred'] = pd.Series(pred)

  if print_stat:
    stat = test_com.groupby('pred')['pred'].count()
    return test_com, stat

  return test_com

def rating(row, class_num):
    count_dict = {}
    for i in range(class_num):
        count_dict[i] = row.count(i)

    count_dict = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)

    if count_dict[0][1] == count_dict[1][1]:
        if class_num == 7:
            return 0
        else:
            return 1
    else:
        return count_dict[0][0]


def voting(result_df_list, class_num=7):
    voting_df = result_df_list[0].copy()
    voting_df = voting_df.rename(columns={'pred': 'pred_1'})

    for i in range(1, len(result_df_list)):
        voting_df['pred_%d' % (i + 1)] = result_df_list[i]['pred']

    column = []
    for i in range(len(result_df_list)):
        column.append('pred_%d' % (i + 1))

    final_pred = []
    for i in range(result_df_list[0].shape[0]):
        final_pred.append(rating(list(voting_df.loc[i, column]), class_num))

    voting_df['pred'] = pd.Series(final_pred)

    return voting_df

def pos_neg_processing(train_df, mode = 'pos'):
  if mode == 'pos':
    new_df = train_df.loc[train_df['original'].isin([1, 2, 3])]
  else:
    new_df = train_df.loc[train_df['original'].isin([4, 5, 6])]
  new_df.drop(['class'], axis = 1, inplace = True)
  new_df['class'] = new_df['original'].copy()
  return new_df

def final_pred(df):
  main = df['main']
  pos = df['pos']
  neg = df['neg']
  if main == 1: return 0
  elif main == 0: return pos
  else: return neg

def mix_df(result_df, pos_df, neg_df):
  result_df = result_df.rename(columns = {'pred' : 'main'})
  result_df['pos'] = pos_df['pred']
  result_df['neg'] = neg_df['pred']
  pred = result_df.apply(final_pred, axis = 1)
  result_df['pred'] = pred
  new_result_df = result_df[['comment', 'original', 'pred']]
  new_result_df = new_result_df.rename(columns = {'original' : 'class'})
  return new_result_df