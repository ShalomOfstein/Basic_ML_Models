import numpy as np
class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X_train, y_train, iterations=1000, alpha=0.0001):
        w = np.random.rand(len(X_train[0]))
        b = np.random.rand()
        for i in range(iterations):
            deriv_b = np.mean(1*((np.dot(X_train,w)+b)-y_train))
            gradient_w = 1.0/len(y_train) * np.dot(((np.dot(X_train,w)+b)-y_train), X_train)
            w -= alpha * gradient_w
            b -= alpha * deriv_b
        self.coefficients = np.concatenate((np.array([b]), w))
        print('Weights:', self.coefficients[1:])
        print('Bias:', self.coefficients[0])

    def predict(self, X):
        return np.dot(X, self.coefficients[1:]) + self.coefficients[0]

    def mean_squared_error(self, y_true, y_pred):
        for i in range(len(y_true)):
            print(f'Actual price: {y_true[i]} , Predicted price: {y_pred[i]}, Difference: {y_true[i] - y_pred[i]}')
        return np.mean((y_true - y_pred) ** 2)