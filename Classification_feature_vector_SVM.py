import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


def decode(datum):
    a = np.zeros((datum.shape[0], 1))
    for j in range(datum.shape[0]):
        a[j] = np.argmax(datum[j])
    return a


def encode(datum):
    return to_categorical(datum)


print("Loading Data ...")
Data = np.loadtxt("/data/alanfr/Desktop/MSc/myAnalysis/results/feature_vector 25.3.2022/data_fin_delta.csv", delimiter=",")

print("Preparing Data ...")
# psd_x = 2049 > [0:2048]
# -------------------------------------
# psd_y = 2049 > [2049:4097]
# -------------------------------------
# mfcc_x_1st = 498 > [4098:4595]
# mfcc_x_2nd = 498 > [4596:5093]
# -------------------------------------
# mfcc_y_1st = 498 > [5094:5591]
# mfcc_y_2nd = 498 > [5592:6089]
# -------------------------------------
# gtcc_x_1st = 498 > [6090:6587]
# gtcc_x_2nd = 498 > [6588:7085]
# gtcc_x_3rd = 498 > [7086:7583]
# gtcc_x_4th = 498 > [7584:8081]
# -------------------------------------
# gtcc_y_1st = 498 > [8082:8579]
# gtcc_y_2nd = 498 > [8580:9077]
# gtcc_y_3rd = 498 > [9078:9575]
# gtcc_y_4th = 498 > [9576:10073]


sub1 = Data[5094:6089, :]
sub2 = Data[8082:10073, :]
res = np.concatenate((sub1, sub2), axis=0)
x = np.transpose(res)

# x = np.transpose(Data[5094:6089, :])

Label_1 = np.transpose(Data[10074:10075, :])
Label_1 = Label_1.astype(int)

Label_2 = np.transpose(Data[10075:10076, :])
Label_2 = Label_2.astype(int)

Label_3 = np.transpose(Data[10076:10077, :])
Label_3 = Label_3.astype(int)

y = encode(Label_3)

np.random.seed(1)
K = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun = 'mse'
optimizer_algorithm = 'adam'
number_inner_layers = 3
number_inner_neurons = 100
number_epoch = 100
batch_length = 10
show_inter_results = 0


accplot = []
cnt = 0
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

for train, test in kfold.split(x, decode(y)):
    cnt = cnt + 1
    print(cnt)

    model = Sequential()

    for i in range(number_inner_layers):
        model.add(Dense(int(number_inner_neurons * 2/3 + y.shape[1]), input_dim=x.shape[1], activation=inner_activation_fun))
    model.add(Dense(y.shape[1], activation=outer_activation_fun))

    model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])
    model.fit(x[train], y[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
    scores = model.evaluate(x[test], y[test], verbose=show_inter_results)

    accplot.append(scores)
    print(scores[1] * 100)
    y_pred = model.predict(x[test])
    np.savetxt("/data/alanfr/Desktop/MSc/myAnalysis/results/feature_vector/results/10_class/Results_3%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')

plt.plot(accplot)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
print(model.summary())
