import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import os


def decode(datum):
    a = np.zeros((datum.shape[0], 1))
    for j in range(datum.shape[0]):
        a[j] = np.argmax(datum[j])
    return a


def encode(datum):
    return to_categorical(datum)


np.random.seed(1)
K = 1
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun = 'mse'
optimizer_algorithm = 'adam'
number_epoch = 10
batch_length = 100
show_inter_results = 1

print("Loading Data ...")
Data = np.loadtxt("/data/alanfr/Desktop/MSc/myAnalysis/results/LSTM_results/x_only/RF_Data.csv", delimiter=",")
# Data = np.loadtxt("/data/alanfr/Desktop/MSc/myAnalysis/LSTM_results/RF_Data.csv", delimiter=",")

print("Preparing Data ...")
x = np.transpose(Data[0:9999, :])
Label_1 = np.transpose(Data[9999:10000, :])
Label_1 = Label_1.astype(int)

Label_2 = np.transpose(Data[10000:10001, :])
Label_2 = Label_2.astype(int)

Label_3 = np.transpose(Data[10001:10002, :])
Label_3 = Label_3.astype(int)

# print("Preparing Data ...")
# x = np.transpose(Data[0:2047, :])
# Label_1 = np.transpose(Data[2048:2049, :])
# Label_1 = Label_1.astype(int)
#
# Label_2 = np.transpose(Data[2049:2050, :])
# Label_2 = Label_2.astype(int)
#
# Label_3 = np.transpose(Data[2050:2051, :])
# Label_3 = Label_3.astype(int)

y = encode(Label_3)
outPutSize = y.shape[1]


cvscores = []
cnt = 0
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

print("Starting teaching process ...")
for train, test in kfold.split(x, decode(y)):

    print('Iteration -- ' + str(cnt/K*100) + ' %')
    cnt = cnt + 1

    anlzNet = y[test]
    x = x.reshape(x.shape[0], x.shape[1], 1)
    outNet = y.reshape(1, -1)[0]

    timeSteps = x.shape[1]
    n_features = 1

    model = Sequential()
    model.add(LSTM(20, input_shape=(timeSteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation=inner_activation_fun))
    model.add(Dense(outPutSize, activation=outer_activation_fun))

    model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])

    print("Fitting Data ...")
    model.fit(x[train], outNet[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
    scores = model.evaluate(x[test], outNet[test], verbose=show_inter_results)

    print('Score == ' + str(scores[1] * 100))
    # cvscores.append(scores[1] * 100)
    y_pred = model.predict(x[test])
    np.savetxt("Results_3%s.csv" % cnt, np.column_stack((anlzNet, y_pred)), delimiter=",", fmt='%s')
