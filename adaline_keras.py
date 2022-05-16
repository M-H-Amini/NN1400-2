############################################################
##          Written By: Mohammad Hossein Amini            ##
##                 Date:  Sat, 04/30/2022                 ##
############################################################


##  Description: Adaline with keras for nerual network TA


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

##  Dataset...
X = np.array([[100, 110, 90, 150, 50]]).T
y = np.array([250, 240, 200, 450, 190])

mean = X.mean()
std = X.std()
XX = (X - mean) / std


##  Visualization...
# plt.figure()
# plt.plot(X, y, "rx")
# plt.show()

##  Model...
opt = SGD(learning_rate=0.03)

model = Sequential()
model.add(Dense(1, activation="linear", input_shape=(1,)))
model.compile(optimizer=opt, loss="mse")

model.fit(XX, y, epochs=500)
# model.summary()

prediction = model.predict(XX)
plt.figure()
plt.plot(X, y, "rx", X, prediction, "bo--")
plt.show()
