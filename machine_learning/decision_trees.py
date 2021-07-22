import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
IMAGE_DIR = 'images/'


def student_drinking_decision_tree(file_name):
    dataset_name = "student_drinking"
    data = pd.read_csv(file_name)
    yy = data["Dalc"]  # only day alcohol data
    xx = data.drop("Dalc", axis=1)  # everything other than day alcohol data
    mse_training_size(dataset_name, yy, xx)

    data_set = train_test_split(xx, yy, test_size=0.4, random_state=3)
    train_X, X_test, train_y, y_test = data_set
    tree_classifier = DecisionTreeClassifier(ccp_alpha=0.0238, criterion='entropy')
    t0 = time.time()
    tree_classifier.fit(train_X, train_y)
    t1 = time.time()
    fit_time = t1 - t0

    train_y_predict = tree_classifier.predict(train_X)
    t0 = time.time()
    y_test_predict = tree_classifier.predict(X_test)
    t1 = time.time()
    test_time = t1 - t0

    # optimize_ccp(dataset_name, tree_classifier, data_set)
    plot_validation_curves(dataset_name, 0.0238, train_X, train_y)
    return accuracy_score(train_y, train_y_predict), accuracy_score(y_test, y_test_predict), fit_time, test_time


def wine_decision_tree(file_name):
    dataset_name = 'wine'
    data = pd.read_csv(file_name)
    yy = data["Class"]  # only class id data
    xx = data.drop("Class", axis=1)  # everything other than class id data
    mse_training_size(dataset_name, yy, xx)

    data_set = train_test_split(xx, yy, test_size=0.2, random_state=3)
    train_X, X_test, train_y, y_test = data_set
    tree_classifier = DecisionTreeClassifier(ccp_alpha=0.082, criterion='entropy')
    t0 = time.time()
    tree_classifier.fit(train_X, train_y)
    t1 = time.time()
    fit_time = t1 - t0

    train_y_predict = tree_classifier.predict(train_X)

    t0 = time.time()

    y_test_predict = tree_classifier.predict(X_test)
    t1 = time.time()
    test_time = t1 - t0


    # optimize_ccp(dataset_name, tree_classifier, data_set)
    plot_validation_curves(dataset_name, 0.082, train_X, train_y)
    return accuracy_score(train_y, train_y_predict), accuracy_score(y_test, y_test_predict), fit_time, test_time


def mse_training_size(dataset_name, y, x):
    name_to_train_size_map = {'wine': [1, 25, 50, 75, 100, 125, 142, 160, 173],
                              'student_drinking': [1, 75, 150, 225, 316]}
    train_sizes = name_to_train_size_map[dataset_name]
    train_sizes, train_scores, validation_scores = learning_curve(tree.DecisionTreeClassifier(random_state=7), x, y,
                                                                  train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Mean Square Error vs Training Size for a Decision Tree model')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_training_size_dt'.format(IMAGE_DIR,dataset_name))


def plot_validation_curves(dataset_name, alpha, train_x, train_y):
    # plot ccp alpha
    parametric_range = [0.002, 0.008, 0.01, 0.015, 0.02, 0.08]
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=7), train_x, train_y,
                                                 param_name="ccp_alpha", param_range=parametric_range)

    trained = np.mean(train_scores, axis=1)
    tested = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(parametric_range, trained, label='Training score')
    plt.plot(parametric_range, tested, label='Cross-validation score')
    plt.title('Validation curve for decision tree')
    plt.xlabel('ccp_alpha value')
    plt.ylabel("Classification score")
    plt.title('classification score vs ccp_alpha for decision trees')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_ccp_alpha_dt.png'.format(IMAGE_DIR, dataset_name))
