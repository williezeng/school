import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

IMAGE_DIR = 'images/'

def student_knn(file_name):
    dataset_name = "student_drinking"
    data = pd.read_csv(file_name)
    yy = data["Dalc"]  # only day alcohol data
    xx = data.drop("Dalc", axis=1)  # everything other than day alcohol data
    data_set = train_test_split(xx, yy, test_size=0.3, random_state=3)
    mse_training_size(dataset_name, yy, xx)
    return search_best_params(data_set, dataset_name)

def wine_knn(file_name):
    dataset_name = 'wine'
    data = pd.read_csv(file_name)
    yy = data["Class"]  # only class id data
    xx = data.drop("Class", axis=1)  # everything other than class id data
    data_set = train_test_split(xx, yy, test_size=0.25, random_state=3)
    mse_training_size(dataset_name, yy, xx)
    return search_best_params(data_set, dataset_name)

def mse_training_size(dataset_name, y, x):

    name_to_train_size_map = {'wine': [1, 25, 50, 75, 100, 125, 142, 160, 173],
                              'student_drinking': [1, 75, 150, 225, 316]}

    name_to_neighbor_map = {'wine': 1, 'student_drinking': 1}

    train_sizes = name_to_train_size_map[dataset_name]
    train_sizes, train_scores, validation_scores = learning_curve(KNeighborsClassifier(n_neighbors=name_to_neighbor_map[dataset_name]), x, y,
                                                                  train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Mean Square Error vs Training Size for knn')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_training_size_knn'.format(IMAGE_DIR, dataset_name))


def search_best_params(data_set, dataset_name):
    train_X, X_test, train_y, y_test = data_set

    param_grid = [
        {
            'n_neighbors': [1, 5, 10, 15, 20, 25],
        }
    ]

    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
    t0 = time.time()
    clf.fit(train_X, train_y)
    t1 = time.time()
    fit_time = t1 - t0

    train_y_predict = clf.predict(train_X)

    t0 = time.time()
    y_test_predict = clf.predict(X_test)
    t1 = time.time()
    test_time = t1 - t0
    print("Best parameters set found on development set:")

    print(clf.best_params_)
    plot_validation_curves(dataset_name, train_X, train_y)

    return accuracy_score(train_y, train_y_predict), accuracy_score(y_test, y_test_predict), fit_time, test_time


def plot_validation_curves(dataset_name, train_x, train_y):
    # plot n_neighbors

    name_to_train_size_map = {'wine': [1, 5, 10, 15, 20, 30, 40, 60, 70],
                              'student_drinking':[1, 5, 10, 20, 30,40,80]}
    train_scores, test_scores = validation_curve(KNeighborsClassifier(), train_x, train_y,
                                                 param_name="n_neighbors", param_range=name_to_train_size_map[dataset_name])

    trained = np.mean(train_scores, axis=1)
    tested = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(name_to_train_size_map[dataset_name], trained, label='Training score')
    plt.plot(name_to_train_size_map[dataset_name], tested, label='Cross-validation score')
    plt.title('Validation curve for knn')
    plt.xlabel('N_neighbors value')
    plt.ylabel("Classification score")
    plt.title('classification score vs learning_rate for knn')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_N_neighbors_knn.png'.format(IMAGE_DIR, dataset_name))
