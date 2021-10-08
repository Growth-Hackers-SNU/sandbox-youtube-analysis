import pandas as pd
import numpy as np

def df_ready(directory='/content/drive/Shareddrives/[GH x Sandbox]/최종분류/sample_final_full.csv', mode = 7):  # 데이터 프레임 준비, 필수적으로 실행해야 함, 디렉토리 지정 가능
    data = pd.read_csv(directory, index_col=0)
    data['class'] = data['class'].astype(int)

    data = data.dropna(axis=0)
    column = []

    if mode == 7:

        data.loc[data['class'] >= 7, 'class'] = 0  # 전체 라벨 사용 경우 7번은 0번 처리
        column = ['comment', 'class']

    elif mode == 3:

        data['original'] = data['class'].copy() # 기존 라벨을 남겨둠

        data.loc[data['class'].isin([1, 2, 3]), 'class'] = 0
        data.loc[data['class'].isin([4, 5, 6]), 'class'] = 2
        data.loc[data['class'] >= 7, 'class'] = 1
        column = ['comment', 'class', 'original']

    else: # full 버전(라벨 개수 동일화 X)

        print("Preprocessing DataFrame Done!")
        data.loc[data['class'] >= 7, 'class'] = 0  # 전체 라벨 사용 경우 7번은 0번 처리
        return data[['comment', 'class']]

    min_class_num = data.groupby(['class'])['index'].count().min()  # 동일 개수 샘플링
    comment_df = pd.DataFrame(columns=['comment', 'class'])

    for i in range(mode):
        comment_df = comment_df.append(data.loc[data['class'] == i, column].sample(min_class_num))

    comment_df = comment_df.reset_index(drop=True)

    if mode == 3:
        comment_df['class'] = comment_df['class'].astype(int)
        comment_df['original'] = comment_df['original'].astype(int)
        comment_df.loc[comment_df['original'] >= 7, 'original'] = 0

    print("Preprocessing DataFrame Done!")

    return comment_df