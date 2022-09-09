import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import vgg19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def decode(datum):
    a = np.zeros((datum.shape[0], 1))
    for index in range(datum.shape[0]):
        a[index] = np.argmax(datum[index])
    return a


def encode(datum):
    return to_categorical(datum)

np.random.seed(1)

print("Loading Data ...")
Source = np.loadtxt("/data/alanfr/Desktop/MSc/myAnalysis/results/CNN - 29.4.2022/data_CNN_fixed.csv", delimiter=",")

Data = Source[0:9996, :]
Target = Source[9996:9999, :]

print("Preparing Data ...")
Data = Data.reshape(9996, 227)

Label_1 = np.transpose(Target[0, :])
Label_1 = Label_1.astype(int)

Label_2 = np.transpose(Target[1, :])
Label_2 = Label_2.astype(int)

Label_3 = np.transpose(Target[2, :])
Label_3 = Label_3.astype(int)

y = encode(Label_1)

cnt = 0
NumClasses = 2
optimizer_loss_fun = 'mse'
optimizer_algorithm = 'adam'
number_epoch = 100
batch_length = 10
show_inter_results = True

x_train, x_test, y_train, y_test = train_test_split(np.transpose(Data), np.transpose(Target), test_size=0.2, random_state=1)
x_train = x_train.reshape(181, 2499, 4)

model = InceptionV3(include_top=False, input_shape=np.shape(x_train), classes=NumClasses, classifier_activation="softmax")
model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
# y_pred = model.predict(x_test)
# print(y_pred)
