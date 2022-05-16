############################################################
##          Written By: Mohammad Hossein Amini            ##
##                 Date:  Sat, 05/07/2022                 ##
############################################################


##  Description: Classification for breast cancer dataset


from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

##  Dataset preparation...
dataset = load_breast_cancer()
data = dataset.data
target = dataset.target

mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std

##  Train-val split...
validation_split = 0.2
indices = np.random.randint(0, len(data), size=(int(validation_split * len(data))))
Xval = data[indices]
yval = target[indices]
# train_indices = [i for i in range(len(data)) if i not in indices]
Xtrain = data[list(set(range(len(data))) - set(indices))]
ytrain = target[list(set(range(len(data))) - set(indices))]

df = pd.DataFrame(data, columns=dataset.feature_names)


# df.plot(kind="hist")
# plt.show()

##  Model creation...
model = Sequential()
model.add(Dense(1, activation="sigmoid", input_shape=(30,)))
# print(model.summary())
model.compile(loss="bce", optimizer="adam", metrics=["acc"])
# model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=50, batch_size=32)
hist = model.fit(data, target, epochs=50, batch_size=32, validation_split=0.3)
loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
acc = hist.history["acc"]
val_acc = hist.history["val_acc"]
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(range(50), loss, "r", range(50), val_loss, "b")
plt.subplot(1, 2, 2)
plt.plot(range(50), acc, "r", range(50), val_acc, "b")
plt.show()
