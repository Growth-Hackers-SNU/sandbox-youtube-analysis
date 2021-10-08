import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def vector_processing(Data):
    ### train, test split 및 머신러닝 모델에 투입 가능한 형태로 벡터를 가공해주는 함수
  column = []
  try:
    a = Data['original']
    column = ['comment', 'original']
  except:
    column = ['comment']
  X = Data['vector']
  y = Data['class']
  com = Data[column]
  vs = []

  for v in X:
    vs.append(np.array(v))

  X = vs
  y = y.astype(int)

  return com, X, y

def df_split(Data, test_size = 0.2):
    ### 학습 전 데이터프레임 전체를 train/test set으로 split하는 함수
    train_df, test_df = train_test_split(Data, test_size=test_size)
    return train_df, test_df

def nb(X_train, y_train, Token = 'kiwi', alpha=1.0, class_prior=None, fit_prior=True):

    ### nb 모델링, vector_processing까지 필수 실행되어야 가능
    # 기타 nb 모델 파라미터인 alpha, class_prior, fit_prior 지정 가능

    # token 받아서 transform 진행하기
  if Token == 'fasttext':
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    transform_vec = scaler.transform(X_train)
  else:
    transform_vec = X_train

  ### 모델에 학습 시키기
  mod = MultinomialNB()
  mod.fit(X_train, y_train)
  MultinomialNB(alpha=alpha, class_prior=class_prior, fit_prior=fit_prior)

  return mod


def svm(X_train, y_train, C=1.0, kernel='linear', gamma='auto'):

    ### svm 모델링, vector_processing까지 필수 실행되어야 가능
    # 기타 svm 모델 파라미터인 C, kernel, gamma 지정 가능
    # 데이터 셋이 크면 소요 시간이 매우 길어진다는 단점

    # 라벨 인코딩 필요
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)

    mod = SVC(C=C, kernel=kernel, gamma=gamma)
    mod.fit(X_train, y_train)

    return mod

########### LSTM #################
import re
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors, FastText
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def eumjul_sentence(texts):
    tokenized = []
    for j in range(len(texts)):
        tokenized.append(texts[j])
    return tokenized

def sen_to_seq(s_list, embedding_model):
    seq = list()
    for s in s_list:
        try:
            seq_value = embedding_model.wv.index2word.index(s)+1
        except:
            seq_value = 0 # 없으면 0
        seq.append(seq_value)
    return seq

def seq_padding(seq, max_len):
    if len(seq) < max_len:
        n = max_len - len(seq)
        zero_list = [0] * n
        seq += zero_list
    elif len(seq) > max_len:
        seq = seq[:max_len]
    return seq

class LSTM_Model():
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    def preprocess(self, df, sentence_len):
        X_test = df['tokenized']
        word_seq_test = X_test.map(lambda x: sen_to_seq(x, self.embedding_model))
        word_seq_test = word_seq_test.map(lambda x : seq_padding(x, sentence_len))
        word_seq_test = np.array(word_seq_test.to_list())
        return word_seq_test

    def predict(self, X_train, mode = 'fasttext_eumjul'):
        print("Preprocessing...")
        if mode == 'fasttext_eumjul':
            x = self.preprocess(X_train, sentence_len=100)
        else: # fasttext
            x = self.preprocess(X_train, sentence_len=15)

        predicted = self.model.predict_classes(x)
        X_train['pred'] = predicted
        return X_train

def lstm(df, num_classes = 7, token = 'eumjul'):
    # Call libraries
    # Set train - validate data
    train_data, test_data = train_test_split(df, test_size=1000, random_state=7607)
    X_train = train_data['tokenized']
    X_test = test_data['tokenized']
    y_train = train_data['class']
    y_test = test_data['class']

    # Load FastText Model
    print("Loading FastText Model...")
    if token == 'eumjul':
        model_fname = "/content/drive/Shareddrives/[GH x Sandbox]/code/DeepLevel/fasttext/fasttext_eumjul"
        embedding_model = FastText.load(model_fname)
    elif token == 'jamo':
        model_fname = "/content/drive/Shareddrives/[GH x Sandbox]/code/DeepLevel/fasttext/jamo_fasttext"
        embedding_model = FastText.load(model_fname)
    else:
        print("token should be eumjul or jamo")
        return 1

    # encoding
    ## set parameter
    max_len = max(len(l) for l in X_train)
    sentence_len = 100 if token == 'eumjul' else 15

    ## make dataframe to train - validate data
    print("Making train - validate data...")
    # print("sen_to_seq")
    word_seq_train = X_train.map(lambda x: sen_to_seq(x, embedding_model))
    word_seq_test = X_test.map(lambda x: sen_to_seq(x, embedding_model))
    # print("seq_padding")
    word_seq_train = word_seq_train.map(lambda x : seq_padding(x, sentence_len))
    word_seq_test = word_seq_test.map(lambda x : seq_padding(x, sentence_len))
    # print("to numpy")
    word_seq_train = np.array(word_seq_train.to_list())
    word_seq_test = np.array(word_seq_test.to_list())

    y_train_catg = to_categorical(y_train, num_classes=num_classes)
    y_test_catg = to_categorical(y_test, num_classes=num_classes)

    # Build Embedding Matrix
    print("Building Embedding Matrix...")
    embedding_matrix = np.zeros((embedding_model.wv.vectors.shape[0]+1, embedding_model.wv.vectors.shape[1]))
    vocab_size = np.shape(embedding_matrix)[0]
    for i in tqdm(range(len(embedding_model.wv.vectors))):
        embedding_matrix[i+1] = embedding_model.wv.vectors[i]

    # Modeling
    ## Build Model and Compile
    print("Building Model...")
    model = Sequential()
    e = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=sentence_len, trainable=False) # input_length = sentence_len
    model.add(e)
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(16, activation = 'tanh'))
    model.add(Dense(num_classes, activation = 'softmax')) # 여기서 num_classes은 최종 라벨 갯수

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    print("Training Model...")
    checkpoint_path = './checkpoints/my_checkpoint2'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=False,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 verbose=1)

    model.fit(word_seq_train, y_train_catg,
              epochs=30, verbose=2,
              validation_data=(word_seq_test, y_test_catg),
              callbacks=[checkpoint])

    model = load_model(checkpoint_path)

    lstm_model = LSTM_Model(model, embedding_model)
    return lstm_model
