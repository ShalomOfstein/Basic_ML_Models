# Machine Learning Models Implementation

This project provides implementations of three machine learning models: Naive Bayes, Linear Regression, and Logistic Regression. The code is organized into separate classes for each model, and a main script demonstrates how to use these models for training and testing on different datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Classes Overview](#classes-overview)
    - [NaiveBayesModel](#naivebayesmodel)
    - [LinearRegression](#linearregression)
    - [LogisticRegression](#logisticregression)
- [Calculations and Methods](#calculations-and-methods)
    - [Training and Prediction](#training-and-prediction)
    - [Evaluation Metrics](#evaluation-metrics)
- [License](#license)

## Installation

To use this project, clone the repository and install the necessary Python packages:

```bash
git clone <repository_url>
cd <repository_directory>
pip install numpy pandas
```
Replace `<repository_url>` and `<repository_directory>` with the actual URL and directory name of your repository.

## Usage

1. **Prepare Your Data**: Place your datasets in the same directory as the script, ensuring they are in CSV format.

2. **Run the Script**: Execute the main script to train and evaluate the models.

   ```bash
   python main.py
   ```
3. **Customize**: Modify the `train_test_split` function or the model parameters in the main script as needed.

## Classes Overview

### NaiveBayesModel

The `NaiveBayesModel` class implements a simple Naive Bayes classifier. It calculates probabilities based on the frequency of feature values within each class.

#### Methods:
- `fit(X, y)`: Trains the model using the provided features `X` and labels `y`.
- `predict(prediction)`: Predicts the class for a new instance or set of instances.
- `score(X_test, y_test)`: Evaluates the model's performance on the test set.

### LinearRegression

The `LinearRegression` class implements a basic linear regression model, fitting a line to the data to predict continuous outcomes.

#### Methods:
- `fit(X_train, y_train, iterations=1000, alpha=0.0001)`: Trains the model using gradient descent.
- `predict(X)`: Predicts the target values for the given feature set `X`.
- `mean_squared_error(y_true, y_pred)`: Calculates the mean squared error for the predictions.

### LogisticRegression

The `LogisticRegression` class implements a logistic regression model, suitable for binary classification problems.

#### Methods:
- `fit(X, y, iterations=1000, alpha=0.0001)`: Trains the model using gradient descent.
- `predict(X)`: Predicts the class labels for the given feature set `X`.
- `score(X_test, y_test)`: Evaluates the model's performance on the test set, returning errors, accuracy, precision, and recall.

## Calculations and Methods

### Training and Prediction

**Naive Bayes**:
- Uses conditional probabilities for each feature given the class label.
- Predicts the class with the highest posterior probability.

**Linear Regression**:
- Uses gradient descent to minimize the mean squared error between the predicted and actual values.

**Logistic Regression**:
- Uses the sigmoid function to map predictions to probabilities.
- Minimizes the cross-entropy loss using gradient descent.

### Evaluation Metrics

- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F-measure**: The harmonic mean of precision and recall.
- **Mean Squared Error (MSE)**: The average of the squared differences between the actual and predicted values.

---

## Additional Information

- **Data Format**: Ensure your datasets are properly formatted and contain no missing values.
- **Customization**: You can modify the hyperparameters, such as the number of iterations or the learning rate, in the main script.
- **Error Handling**: Implement additional checks and error handling as needed, especially when dealing with different data types or missing values.


Feel free to collaborate and contribute to the improvement of this project!
