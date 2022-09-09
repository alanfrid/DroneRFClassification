import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, InputLayer, Dense, TimeDistributed, Conv2D, MaxPool2D, Flatten, LSTM, Dropout


def decode(datum):
    a = np.zeros((datum.shape[0], 1))
    for j in range(datum.shape[0]):
        a[j] = np.argmax(datum[j])
    return a


def encode(datum):
    return to_categorical(datum)


np.random.seed(1)
K = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun = 'mse'
optimizer_algorithm = 'adam'
number_inner_layers = 3
number_inner_neurons = 256
number_epoch = 1
batch_length = 1
show_inter_results = 0
verbose = 0

print("Loading Data ...")
Data = np.loadtxt(r"C:\users\alan9\Desktop\Msc\Research\My Analysis\DroneRF-master\Python\RF_Data_short.csv", delimiter=",")

# print("Preparing Data ...")
# x = np.transpose(Data[0:10000, :])
# Label_1 = np.transpose(Data[10000:10001, :])
# Label_1 = Label_1.astype(int)
#
# Label_2 = np.transpose(Data[10001:10002, :])
# Label_2 = Label_2.astype(int)
#
# Label_3 = np.transpose(Data[10002:10003, :])
# Label_3 = Label_3.astype(int)
#
# outputType = [2, 4, 10]
# y = encode(Label_2)

print("Preparing Data ...")
x = np.transpose(Data[0:7, :])
Label_1 = np.transpose(Data[7:8, :])
Label_1 = Label_1.astype(int)

Label_2 = np.transpose(Data[8:9, :])
Label_2 = Label_2.astype(int)

Label_3 = np.transpose(Data[9:10, :])
Label_3 = Label_3.astype(int)

outputType = [2, 4, 10]
y = encode(Label_2)

cvscores = []
cnt = 0
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

print("Starting teaching process ...")
for train, test in kfold.split(x, decode(y)):

    print('Iteration index -- ' + str(cnt/len(train)*100) + ' %')
    cnt = cnt + 1

    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(1, -1)[0]

    timesteps = x.shape[1]
    n_features = 1
    model = Sequential()
    model.add(LSTM(100, input_shape=(timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])

    model.fit(x[train], y[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
    scores = model.evaluate(x[test], y[test], verbose=show_inter_results)

    print(scores[1] * 100)
    cvscores.append(scores[1] * 100)
    y_pred = model.predict(x[test])
    np.savetxt("Results_2%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')

