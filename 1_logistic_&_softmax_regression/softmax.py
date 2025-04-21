import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
load_dotenv()

def init_params():
    rng = np.random.default_rng(seed=1)
    w = rng.random(size=(3, 4))
    b = np.zeros((3, 1))
    return w, b

def softmax(z):
    z = z.astype(float)
    eps = 1e-8
    a = np.exp(z + eps) / (np.sum(np.exp(z + eps), axis=0, keepdims=True))
    return a

def forward(w, x, b):
    z = np.dot(w, x) + b
    a = softmax(z)
    return z, a

def one_hot(y):
    y_onehot = np.zeros((int(np.max(y) + 1), y.size)) 
    y_onehot[y.astype(int), np.arange(y.size)] = 1
    return y_onehot

def CCE(a, y_onehot):
    eps = 1e-8
    loss = - np.sum(y_onehot * np.log(a + eps)) / y_onehot.shape[1]
    return loss

def accuracy(y, a):
    pred = np.argmax(a, axis=0, keepdims=True)
    acc = np.sum(y == pred) / y.size * 100
    return acc

def backward(y_onehot, a, x):
    dz = a - y_onehot
    dw = np.dot(dz, x.T) / y_onehot.shape[1]
    db = np.sum(dz, axis=1, keepdims=True) / y_onehot.shape[1]
    return dw, db

def update(w, b, alpha, dw, db):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def gradient_descent(x, y, epochs, w, b, alpha):
    y_onehot = one_hot(y)
    for epoch in range(epochs):
        z, a = forward(w, x, b)
        
        loss = CCE(a, y_onehot)
        acc = accuracy(y, a)

        dw, db = backward(y_onehot, a, x)
        w, b = update(w, b, alpha, dw, db)
        
        if epoch % 1000 == 0:
            print(f"EPOCH: {epoch}")
            print(f"ACCURACY: {acc}%")
            print(f"LOSS: {loss}\n")
    
    return w, b

def model(x, y, epochs, alpha):
    w, b = init_params()
    w, b = gradient_descent(x, y, epochs, w, b, alpha)
    return w, b

if __name__ == "__main__":
    local_dir = os.getenv("LOCAL_PATH")
    data = pd.read_csv(f"{local_dir}\datasets\iris.csv")
    data = np.array(data)

    X_train = data[:, 0:4].T
    Y_train = data[:, 4]

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train).reshape(1, -1)
    # print(Y_train.shape)
    w, b = model(X_train, Y_train, 10000, 0.01)