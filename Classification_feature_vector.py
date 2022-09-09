import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn import svm
# from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


def decode(datum):
    a = np.zeros((datum.shape[0], 1))
    for index in range(datum.shape[0]):
        a[index] = np.argmax(datum[index])
    return a


def encode(datum):
    return to_categorical(datum)


def sublists(l):
    subs = [[]]
    for ii in range(len(l) + 1):
        for jj in range(ii):
            subs.append(l[jj: ii])
    return subs


print("Loading Data ...")
data_1 = np.loadtxt(r"C:\Users\alan9\Desktop\Msc\data_fin1.csv", delimiter=",")
data_2 = np.loadtxt(r"C:\Users\alan9\Desktop\Msc\data_fin2.csv", delimiter=",")

print("Loading done, preparing Data ...")

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

subFeatures = sublists(['[0:4001]',  # psd_x
                        '[2049:4097]',  # psd_y
                        '[4098:4595]', '[4596:5093]',  # mfcc_x
                        '[5094:5591]', '[5592:6089]',  # mfcc_y
                        '[6090:6587]', '[6588:7085]', '[7086:7583]', '[7584:8081]',  # gtcc_x
                        '[8082:8579]', '[8580:9077]', '[9078:9575]', '[9576:10073]'])  # gtcc_y

subFeaturesNames = sublists(['psd_x',  # psd_x
                             'psd_y',  # psd_y
                             'mfcc_x_coeff_1', 'mfcc_x_coeff_2',  # mfcc_x
                             'mfcc_y_coeff_1', 'mfcc_y_coeff_2',  # mfcc_y
                             'gtcc_x_coeff_1', 'gtcc_x_coeff_2', 'gtcc_x_coeff_3', 'gtcc_x_coeff_4',  # gtcc_x
                             'gtcc_y_coeff_1', 'gtcc_y_coeff_2', 'gtcc_y_coeff_3', 'gtcc_y_coeff_4'])  # gtcc_y

np.random.seed(1)

classification_classes = 1
K = 10
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

for d in range(2):  # --> Iterate through the two dataset types

    # [?] Choose which dataset to use
    if d == 0:
        print("--> Data -- Coefficients")
        Data = data_1
        Label_1 = np.transpose(Data[10074:10075, :])
        Label_1 = Label_1.astype(int)

        Label_2 = np.transpose(Data[10075:10076, :])
        Label_2 = Label_2.astype(int)

        Label_3 = np.transpose(Data[10076:10077, :])
        Label_3 = Label_3.astype(int)
    else:
        print("--> Data -- Deltas")
        Data = data_2
        Label_1 = np.transpose(Data[10074:10075, :])
        Label_1 = Label_1.astype(int)

        Label_2 = np.transpose(Data[10075:10076, :])
        Label_2 = Label_2.astype(int)

        Label_3 = np.transpose(Data[10076:10077, :])
        Label_3 = Label_3.astype(int)

    # [?] Choose what classification type to do
    if classification_classes == 1:
        classification = Label_1
    elif classification_classes == 2:
        classification = Label_2
    elif classification_classes == 3:
        classification = Label_3

    print("Classifying ...\n")
# for r in [0.5, 0.3, 0.1, 0.05, 0.01]:     # --> Set sizes of test data
    # print("Test size = " + str(r * 100) + "%")
    maxLinearResult = 0
    maxRbfResult = 0
    resultsIndexMatrix_Linear = []
    resultsIndexMatrix_RBF = []
    subFeatures = subFeatures[1:len(subFeatures)]
    subFeaturesNames = subFeaturesNames[1:len(subFeaturesNames)]
    for i in range(len(subFeatures)):   # --> Go over all combinations of the dataset features

        avgScoreLinear = 0
        avgScoreRBF = 0
        wholeDomain = subFeatures[i]
        result = [[]]
        # print("Training data domain = " + str(wholeDomain))

        # [?] Extract limits of data sections
        for j in range(len(wholeDomain)):
            thisDomain = wholeDomain[j]
            start = int(thisDomain[thisDomain.index('[') + 1:thisDomain.index(':')])
            end = int(thisDomain[thisDomain.index(':') + 1:thisDomain.index(']')])
            sub = Data[start:end, :]
            if j == 0:
                result = sub
            else:
                result = np.concatenate((result, sub), axis=0)
        x = np.transpose(result)

        # 2. Split data to training and testing
        # TODO: add KFold
        # x_train, x_test, y_train, y_test = train_test_split(x, classification, test_size=r, random_state=109)
        for train, test in kfold.split(x, classification):
            y = classification
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]

            # 3.1 LINEAR SVM
            model_1 = svm.SVC(kernel="linear")
            model_1.fit(x_train, np.reshape(y_train, (len(y_train),)))
            y_pred_1 = model_1.predict(x_test)

            # 3.2 Radial Basis Function (RBF) SVM
            model_2 = svm.SVC(kernel="rbf")
            model_2.fit(x_train, np.reshape(y_train, (len(y_train),)))
            y_pred_2 = model_2.predict(x_test)

            # 4. Calculate Accuracy
            linearResult = metrics.accuracy_score(y_test, y_pred_1) * 100
            rbfResult = metrics.accuracy_score(y_test, y_pred_2) * 100

            avgScoreLinear += linearResult/K
            avgScoreRBF += rbfResult/K

        # 5. Find max result
        resultsIndexMatrix_Linear.append(avgScoreLinear)
        resultsIndexMatrix_RBF.append(avgScoreRBF)

    maxLinearResult = max(resultsIndexMatrix_Linear)
    maxRbfResult = max(resultsIndexMatrix_RBF)
    print("--> Max-Avg Linear SVM == " + str(maxLinearResult))
    print("--> Max-Avg RBF SVM == " + str(maxRbfResult))

    indices = [q for q, w in enumerate(resultsIndexMatrix_Linear) if w == maxLinearResult]
    print("--> Subsets of maximum-average accuracy:")
    print([subFeaturesNames[e] for e in indices])
    print()
print("Done.")
