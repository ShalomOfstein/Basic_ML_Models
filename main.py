import pandas as pd
import numpy as np
from NaiveBayes import NaiveBayesModel
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression


def train_test_split(data, test_size=0.2,seed=53):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def print_scores(errors, accuracy, precision, recall):
    print(f"Errors: {errors}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

################################ Naive Bayes ################################
print("Naive Bayes\n")
naive_bayes_data = pd.read_csv('naive_bayes_data.csv')

Train, Test = train_test_split(naive_bayes_data, 0.8)

X_train = Train.iloc[:, :-1].values
y_train = Train.iloc[:, -1].values
X_test = Test.iloc[:, :-1].values
y_test = Test.iloc[:, -1].values

model = NaiveBayesModel()
model.fit(X_train, y_train)
errors, accuracy, precision, recall = model.score(X_test, y_test)
print_scores(errors, accuracy, precision, recall)

# new_instance = [['young', 'medium']]
new_instance = pd.DataFrame({'Age': ['young'],'Income': 'medium'})
print("new instance:\n", new_instance)
print("Prediction for new instance:", model.predict(new_instance))


################################ Linear Regression ################################
print("\nLinear Regression\n")

LinearRegressionData = pd.read_csv("prices.txt")
Train, Test = train_test_split(LinearRegressionData, 0.25)

X_train = Train.iloc[:, :-1].values
y_train = Train.iloc[:, -1].values.flatten()
X_test = Test.iloc[:, :-1].values
y_test = Test.iloc[:, -1].values.flatten()

model = LinearRegression()
model.fit(X_train, y_train, iterations=1000, alpha=0.0001)
predictions = model.predict(X_test)
mse = model.mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

################################ Logistic Regression ################################
print("\nLogistic Regression\n")

LogisticRegressionData = pd.read_csv("prices.txt")

Train, Test = train_test_split(LogisticRegressionData, 0.15, seed=54)

X_train = Train.drop(columns=Train.columns[-2]).values
y_train = Train.iloc[:, -2].values.flatten()
X_test = Test.drop(columns=Test.columns[-2]).values
y_test = Test.iloc[:, -2].values

model = LogisticRegression()
model.fit(X_train, y_train, iterations=1000, alpha=0.0001)
predictions = model.predict(X_test)
errors, accuracy, precision, recall = model.score(X_test, y_test)
print_scores(errors, accuracy, precision, recall)
print("F-measure: ", (2 * precision * recall) / (precision + recall))





