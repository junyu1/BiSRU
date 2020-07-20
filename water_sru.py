from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Embedding, Input,Activation
from keras.layers import LSTM,GRU,Bidirectional
from keras.datasets import imdb

from sru import SRU

import numpy as np 
import pandas as pd
from sklearn.preprocessing import Normalizer

import datetime
from evaluate import *
from collections import Counter

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  
sess = tf.Session(config=config)
KTF.set_session(sess) 

batch_size = 128

for depth in range(1,16) :

    print('Loading data...')

    traindata = pd.read_csv('datasets/water/train.csv', header=None)
    testdata = pd.read_csv('datasets/water/test.csv', header=None)

    X = traindata.iloc[: ,:-8]
    Y = traindata.iloc[:,-8:]
    X_test = testdata.iloc[:,:-8]
    Y_test = testdata.iloc[:,-8:]

    X = np.array(X)
    X_test = np.array(X_test)


    scaler = Normalizer().fit(X)
    trainX = scaler.transform(X)
    np.set_printoptions(precision=3) 
    scaler = Normalizer().fit(X_test)
    testX = scaler.transform(X_test)
    np.set_printoptions(precision=3)

    X_train = np.reshape(trainX, (trainX.shape[0], 1,trainX.shape[1]))
    X_test = np.reshape(testX, (testX.shape[0],  1,testX.shape[1]))

    y_train = np.array(Y)
    y_train.flatten()

    y_test = np.array(Y_test)
    y_test.flatten()


    print(X_train)
    print(y_train)

    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)


    print('Build model...')


    model = Sequential()
    if depth<2 :
        model.add(Bidirectional(SRU(128,return_sequences=False,dropout=0.1),input_shape=(1,X_train.shape[2])))
        model.add(Activation('relu'))
    else :
        model.add(Bidirectional(SRU(128,return_sequences=True,dropout=0.1),input_shape=(1,X_train.shape[2])))
        model.add(Activation('relu'))
        for i in range(0,depth-2):
            model.add(Bidirectional(SRU(128,return_sequences=True,dropout=0.1)))
            model.add(Activation('relu'))
        model.add(Bidirectional(SRU(128,return_sequences=False,dropout=0.1)))
        model.add(Activation('relu'))
       model.add(Dense(8,activation='softmax'))
    
    model.summary()
    

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', precision, recall, f1])

    print('Train...')

    starttime = datetime.datetime.now()

    model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=8,
          validation_data=(X_test, y_test))
    endtime = datetime.datetime.now()

    print('total time:'+str((endtime - starttime).seconds))

    print(model.evaluate(X_test, y_test))


    p = model.predict_classes(X_test,verbose=1)

    print(y_test)
    print(np.argmax(y_test,axis=1))

    with open('result/water/BiSRU_depth='+str(depth)+'/time.csv','w') as f6:
        f6.write(str((endtime - starttime).seconds))
        
    np.savetxt('result/water/BiSRU_depth='+str(depth)+'/result.csv',np.argmax(y_test,axis=1),delimiter=',',fmt='%d')
    np.savetxt('result/water/BiSRU_depth='+str(depth)+'/predict.csv',p,delimiter=',',fmt='%d')
    
