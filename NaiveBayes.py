import pandas as pd
import numpy as np


class NaiveBayesModel:

    def __init__(self):
        self.class_prob = {}
        self.conditional_prob = {}
        self.classes = []
        self.alpha = 1

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            self.class_prob[c] = np.sum(y == c) / float(len(y))
            self.conditional_prob[c] = {}
            for i in range(X.shape[1]):
                self.conditional_prob[c][i] = {}
                for val in np.unique(X[:, i]):
                    self.conditional_prob[c][i][val] = (np.sum((X[:, i] == val) & (y == c)) + self.alpha) / (
                            np.sum(y == c) + self.alpha * len(np.unique(X[:, i])))

    def predict(self, prediction):
        if isinstance(prediction, pd.DataFrame):
            prediction = prediction.values.tolist()
        predictions = []
        for x in prediction:
            probs = []
            for c in self.classes:
                prob = self.class_prob[c]
                for i in range(len(x)):
                    feature_val = x[i]
                    if feature_val in self.conditional_prob[c][i]:
                        prob *= self.conditional_prob[c][i][feature_val]
                probs.append(prob)
            predictions.append(self.classes[np.argmax(probs)])
        return predictions

    def score(self, X_test, y_test):
        errors = {"True Positive": 0, "True Negative": 0, "False Positive": 0, "False Negative": 0}
        predictions = self.predict(X_test)

        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                if predictions[i] == "Yes":
                    errors["True Positive"] += 1
                else:
                    errors["True Negative"] += 1
            else:
                if predictions[i] == "Yes":
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
