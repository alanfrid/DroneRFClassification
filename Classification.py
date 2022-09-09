import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold


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
number_epoch = 100
batch_length = 10
show_inter_results = 0

print("Loading Data ...")
Data = np.loadtxt("D:\DroneRF\Data\oldData\RF_Data_Bebop_Phantom.csv", delimiter=",")

print("Preparing Data ...")
x = np.transpose(Data[0:2047, :])
Label_1 = np.transpose(Data[2048:2049, :])
Label_1 = Label_1.astype(int)

Label_2 = np.transpose(Data[2049:2050, :])
Label_2 = Label_2.astype(int)

Label_3 = np.transpose(Data[2050:2051, :])
Label_3 = Label_3.astype(int)

y = encode(Label_2)

cvscores = []
cnt = 0
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

for train, test in kfold.split(x, decode(y)):
    cnt = cnt + 1
    print(cnt)

    model = Sequential()
    for i in range(number_inner_layers):
        model.add(Dense(int(number_inner_neurons / 2), input_dim=x.shape[1], activation=inner_activation_fun))
    model.add(Dense(y.shape[1], activation=outer_activation_fun))
    model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])

    model.fit(x[train], y[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
    scores = model.evaluate(x[test], y[test], verbose=show_inter_results)

    print(scores[1] * 100)
    cvscores.append(scores[1] * 100)
    y_pred = model.predict(x[test])
    np.savetxt("Results_2%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')
