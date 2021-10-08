def nb(Data, Token = 'kiwi', class_num = 7, alpha=1.0, class_prior=None, fit_prior=True):

    ### nb 모델링, vector_ready까지 필수 실행되어야 가능
    # class_num = 3 과 7 중 선택
    # 기타 nb 모델 파라미터인 alpha, class_prior, fit_prior 지정 가능
    # 정확도와 loss를 출력하고 예측 라벨와 실제 라벨을 반환
    try:
        if class_num is not in [3, 7]:
        raise Exception
    except:
        print('Wrong class_num')

    if class_num == 7:
        X_train = Data['vector']
        y_train = Data['class']
    else
        X_train = Data['vector']
        y_train = Data['pos_neg_neu']

    ### token 받아서 transform 진행하기
    if Token == 'fasttext':
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        transform_vec = scaler.transform(X_train)
    else:
        transform_vec = X_train


  ### 모델에 학습 시키기
    mod = MultinomialNB()
    mod.fit(transform_vec, y_train.astype(int))
    MultinomialNB(alpha=alpha, class_prior=class_prior, fit_prior=fit_prior)

    return mod