import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics


def decode(datum):
    a = np.zeros((datum.shape[0], 1))
    for j in range(datum.shape[0]):
        a[j] = np.argmax(datum[j])
    return a

np.random.seed(1)

show_inter_results = 0

print("Loading Data ...")
Data = np.loadtxt(r"C:\Users\alan9\Desktop\Msc\Research\Analysis work\Results\only_GTCC\GTCC_feature_vector.csv", delimiter=",")

print("Preparing Data ...")
x = np.transpose(Data[0:19988, :])
Label_1 = np.transpose(Data[19988:19989, :])
Label_1 = Label_1.astype(int)

Label_2 = np.transpose(Data[19989:19990, :])
Label_2 = Label_2.astype(int)

Label_3 = np.transpose(Data[19990:19991, :])
Label_3 = Label_3.astype(int)

y = Label_1

cvscores = []
cnt = 0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

model = svm.SVC(kernel='rbf')
model.fit(x_train, np.reshape(y_train, (len(y_train), )))

y_pred = model.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# np.savetxt("Results_2%s.csv" % cnt, np.column_stack((y_test, y_pred)), delimiter=",", fmt='%s')
