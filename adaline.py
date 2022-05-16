import numpy as np
import matplotlib.pyplot as plt

##  Dataset...
X = np.array([100, 120, 80, 150, 50])[:, np.newaxis]
y = np.array([400, 500, 300, 410, 200])[:, np.newaxis]

##  Visualize...
# plt.figure()
# plt.plot(X, y, "rx")
# plt.title("Housing Price Dataset")
# plt.xlabel("Size")
# plt.ylabel("Price")
# plt.show()

##  Model...
class Adaline:
    def __init__(self, input_dim):
        print("Adaline created!!")
        self.input_dim = input_dim
        self.w = np.zeros((input_dim + 1, 1))

    def predict(self, x):
        return np.dot(x, self.w)

    def fit(self, X, y, epochs=100, learning_rate=0.1):
        m = len(X)
        XX = np.concatenate((np.ones((m, 1)), X), axis=1)

        for _ in range(epochs):
            a = self.predict(XX)
            error = (y - a).T
            s = np.dot(error, XX).T
            self.w = self.w + learning_rate / m * s
            print(f"epoch: {_} \tw: ", self.w)


adaline = Adaline(1)
print(adaline.predict([[1, 120]]))
adaline.fit(X, y, 100, 0.00001)

XX = np.concatenate((np.ones((len(X), 1)), X), axis=1)
outputs = adaline.predict(XX)

plt.figure()
plt.plot(X, y, "rx", X, outputs, "b-o")
plt.show()
