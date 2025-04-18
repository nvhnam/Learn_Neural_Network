import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os

load_dotenv()

def init_params():
    rng = np.random.default_rng(seed=1)
    w = rng.normal(size=(1,2))
    b = np.zeros((1, 1))
    return w, b

def sigmoid(z):
    eps = 1e-10
    a = 1 / (1 + np.exp(-z + eps))
    return a

def forward(x, w, b):
    z = np.dot(w, x) + b
    a = sigmoid(z)
    return z, a

def BCE(y, a):
    eps = 1e-8
    loss = np.mean(-y * np.log(a + eps) + (1 - y) * np.log((1 - a) + eps))
    return loss

def backward(x, y, a):
    dw = np.dot((a - y), x.T) / y.size  
    db = np.sum(a - y) / y.size
    return dw, db

def update(w, b, alpha, dw, db):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def acc(y, a):
    pred = np.around(a).astype(int)
    accuracy = np.mean(y == pred) * 100 
    return accuracy

def gradient_descent(x, y, alpha, w, b, epochs):
    for epoch in range(epochs):
        z, a = forward(x, w, b)
        
        loss = BCE(y, a)
        accuracy = acc(y, a)

        dw, db = backward(x, y, a)
        w, b = update(w, b, alpha, dw, db)

        if epoch % 1000 == 0:
            print(f"EPOCH: {epoch}")
            print(f"ACCURACY: {accuracy}%")
            print(f"LOSS: {loss}\n")
    
    return w, b

def model(x, y, alpha, epochs):
    w, b = init_params()
    w, b = gradient_descent(x, y, alpha, w, b, epochs)
    return w, b

if __name__ == "__main__":
    local_dir = os.getenv("LOCAL_PATH")
    data = pd.read_csv(f"{local_dir}\datasets\logistic_regression_dataset.csv")
    data = np.array(data)

    X_train = data[:, 0:-1].T
    Y_train = data[:, -1].reshape(1, -1)

    w, b = model(X_train, Y_train, 0.01, 100000)
    