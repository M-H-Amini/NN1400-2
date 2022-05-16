############################################################
##          Written By: Mohammad Hossein Amini            ##
##                 Date:  Mon, 05/16/2022                 ##
############################################################


##  Description: SLP and MLP for MNIST

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

##  Data preparation...

(Xtrain, ytrain), (Xtest, ytest) = load_data()
Xtrain = Xtrain / 255.
Xtest = Xtest / 255.

def show(X, y):
    indices = np.random.randint(len(X), size=(16,))
    plt.figure()
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i*4+j+1)
            plt.imshow(X[indices[i*4+j]])
            plt.title(str(y[indices[i*4+j]]))
            plt.axis('off')
    plt.show()

# show(Xtrain, ytrain)

# Xtrain = Xtrain.reshape((len(Xtrain), 784))
# Xtest = Xtest.reshape((len(Xtest), 784))
# ytemp = np.zeros((len(Xtrain), 10))
# ytemp[range(len(Xtrain)), ytrain] = 1
# ytrain = ytemp.copy()
# ytemp = np.zeros((len(Xtest), 10))
# ytemp[range(len(Xtest)), ytest] = 1
# ytest = ytemp.copy()

## Single-Layer Neural Network...

model = Sequential()
model.add(InputLayer((28, 28)))
model.add(Flatten())
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(Xtrain, ytrain, epochs=10, batch_size=16, validation_data=(Xtest, ytest))

