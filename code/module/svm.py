def svm(Data, class_num=7, C=1.0, kernel='linear', gamma='auto'):
    ### svm 모델링, vector_ready까지 필수 실행되어야 가능
    # class_num = 3 과 7 중 선택
    # 기타 svm 모델 파라미터인 C, kernel, gamma 지정 가능
    # 정확도와 loss를 출력하고 예측 라벨와 실제 라벨을 반환
    # 데이터 셋이 크면 소요 시간이 매우 길어진다는 단점

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

    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)

    SVM = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    SVM.fit(X_train, y_train)

    return SVM