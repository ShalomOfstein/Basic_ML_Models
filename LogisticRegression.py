import numpy as np
import pandas as pd

class LogisticRegression :
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def h(self, X, w, b):
        return self.sigmoid(np.dot(X, w) + b)

    def fit(self, X, y, iterations=1000, alpha=0.0001):
        self.w = np.random.rand(len(X[0]))
        self.b = np.random.rand()
        alpha = 0.001
        for _ in range(10000):
            deriv_b = np.mean(1 * (self.h(X, self.w, self.b) - y))
            gradient_w = 1.0 / len(y) * np.dot((self.h(X, self.w, self.b) - y), X)
            self.b -= alpha * deriv_b
            self.w -= alpha * gradient_w

        print('Weights:', self.w)
        print('Bias:', self.b)

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_predicted = self.sigmoid(linear_model)
        # print("y_predicted", y_predicted)
        return [1 if i > 0.5 else 0 for i in y_predicted]

    def score(self, X_test, y_test):
        errors = {"True Positive": 0, "True Negative": 0, "False Positive": 0, "False Negative": 0}
        predictions = self.predict(X_test)

        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                if predictions[i] == 1:
                    errors["True Positive"] += 1
                else:
                    errors["True Negative"] += 1
            else:
                if predictions[i] == 1:
                    errors["False Positive"] += 1
                else:
                    errors["False Negative"] += 1

        accuracy = (errors["True Positive"] + errors["True Negative"]) / sum(errors.values())
        precision = errors["True Positive"] / (errors["True Positive"] + errors["False Positive"]) if (errors[
                                                                                                           "True Positive"] +
                                                                                                       errors[
                                                                                                           "False Positive"]) > 0 else 0
        recall = errors["True Positive"] / (errors["True Positive"] + errors["False Negative"]) if (errors[
                                                                                                        "True Positive"] +
                                                                                                    errors[
                                                                                                        "False Negative"]) > 0 else 0
        return errors, accuracy, precision, recall
