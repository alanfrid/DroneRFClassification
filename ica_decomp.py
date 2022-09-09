from sklearn.decomposition import FastICA,PCA
import matplotlib.pyplot as plt
import numpy as np

print("Loading Data ...")
source_1 = np.loadtxt("/data/alanfr/Desktop/MSc/myAnalysis/All/10001L_16.csv", delimiter=","); source_reshaped_1 = np.reshape(source_1, (len(source_1), 1))
source_2 = np.loadtxt("/data/alanfr/Desktop/MSc/myAnalysis/All/00000L_17.csv", delimiter=","); source_reshaped_2 = np.reshape(source_2, (len(source_2), 1))
# source_3 = np.loadtxt("/data/alanfr/Desktop/MSc/myAnalysis/All/00000L_18.csv", delimiter=","); source_reshaped_3 = np.reshape(source_3, (len(source_3), 1))
# source_4 = np.loadtxt("/data/alanfr/Desktop/MSc/myAnalysis/All/00000L_19.csv", delimiter=","); source_reshaped_4 = np.reshape(source_4, (len(source_4), 1))

signal = source_reshaped_1

ica_1 = PCA()
S = ica_1.fit(signal).transform(signal)

plt.plot(S)
plt.show()

# ica_2 = fastica(source_reshaped_2)
# ica_3 = fastica(source_reshaped_3)
# ica_4 = fastica(source_reshaped_4)

# print(str(ica_1[0]) + " " + str(ica_1[1]))
# print(str(ica_2[0]) + " " + str(ica_2[1]))
# print(str(ica_3[0]) + " " + str(ica_3[1]))
# print(str(ica_4[0]) + " " + str(ica_4[1]))

# Noise
# [[-9.18431954e-05]] [[1.]]
# [[-0.00011327]] [[1.]]
# [[7.33356591e-05]] [[-1.]]
# [[-7.74634247e-05]] [[1.]]

# Bebop
# [[-5.34474105e-05]] [[-1.]]
# [[-6.09338851e-05]] [[1.]]
# [[4.24633078e-05]] [[-1.]]
# [[5.54956824e-05]] [[-1.]]

# AR
# [[6.57803979e-05]] [[-1.]]
# [[-5.54314113e-05]] [[-1.]]
# [[6.09964856e-05]] [[-1.]]
# [[-7.08620602e-05]] [[-1.]]

# Phantom
# [[8.46974681e-06]] [[1.]]
# [[-8.38996256e-06]] [[1.]]
# [[7.68225771e-06]] [[1.]]
# [[7.09945895e-06]] [[1.]]