import time
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
IMAGE_DIR = 'images/'

@ignore_warnings(category=ConvergenceWarning)
def wine_nn(file_name):
    dataset_name = "wine"
    data = pd.read_csv(file_name)
    yy = data["Class"]  # only class id data
    xx = data.drop("Class", axis=1)  # everything other than class id data
    data_set = train_test_split(xx, yy, test_size=0.15, random_state=3)
    train_X, X_test, train_y, y_test = data_set
    mse_training_size(dataset_name, yy, xx)

    plot_validation_curves(dataset_name, train_X, train_y)
    return search_best_params(data_set, dataset_name)


@ignore_warnings(category=ConvergenceWarning)
def student_drinking_nn(file_name):
    dataset_name = "student_drinking"
    data = pd.read_csv(file_name)
    yy = data["Dalc"]  # only day alcohol data
    xx = data.drop("Dalc", axis=1)  # everything other than day alcohol data
    data_set = train_test_split(xx, yy, test_size=0.2, random_state=3)
    train_X, X_test, train_y, y_test = data_set
    mse_training_size(dataset_name, yy, xx)

    plot_validation_curves(dataset_name, train_X, train_y)
    return search_best_params(data_set, dataset_name)

def search_best_params(data_set, dataset_name=None):
    train_X, X_test, train_y, y_test = data_set

    param_grid = [
        {
            'beta_1': [0.50, 0.66, 0.75, 0.95, 0.99],
            'max_iter': [75, 100, 200, 400, 800, 900, 1000],
            'hidden_layer_sizes': [
                (5,), (20,), (40,), (60,), (80,),(100,),(200,)
            ]
        }
    ]

    clf = GridSearchCV(MLPClassifier(alpha=0.002, random_state=7), param_grid, cv=3,
                       scoring='accuracy')
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
    return accuracy_score(train_y, train_y_predict), accuracy_score(y_test, y_test_predict), fit_time, test_time

def mse_training_size(dataset_name, y, x):
    # get mse vs training size graph
    name_to_train_size_map = {'wine': [1, 25, 50, 75, 100, 125, 142, 173],
                              'student_drinking': [1, 75, 150, 225, 316]}

    nn_settings = {'wine': {'beta_1':0.75, 'hidden_layer_sizes':100,'max_iter':800}, 'student_drinking': {'beta_1': 0.75, 'hidden_layer_sizes': (60,), 'max_iter': 200}}
    train_sizes = name_to_train_size_map[dataset_name]
    train_sizes, train_scores, validation_scores = learning_curve(MLPClassifier(beta_1=nn_settings[dataset_name]['beta_1'],
                                                                                hidden_layer_sizes=(nn_settings[dataset_name]['hidden_layer_sizes']),
                                                                                max_iter=nn_settings[dataset_name]['max_iter'],
                                                                                alpha=0.002, random_state=7), x, y,
                                               train_sizes=train_sizes, cv=5,
                                               scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Mean Square Error vs Training size for a Neural Network model')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_training_size_nn'.format(IMAGE_DIR,dataset_name))

def plot_validation_curves(dataset_name, train_x, train_y):
    # plot ccp alpha
    name_to_train_size_map = {'wine': [0.0001, 0.001, 0.002, 0.005, 0.008, 0.01],
                              'student_drinking': [0.0001, 0.002,0.008,0.01,0.015,0.02,0.08]}
    train_scores, test_scores = validation_curve(MLPClassifier(beta_1=0.75, hidden_layer_sizes=(100), max_iter=800, random_state=7), train_x, train_y,
                                                 param_name="alpha", param_range=name_to_train_size_map[dataset_name])

    trained = np.mean(train_scores, axis=1)
    tested = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(name_to_train_size_map[dataset_name], trained, label='Training score')
    plt.plot(name_to_train_size_map[dataset_name], tested, label='Cross-validation score')
    plt.title('Validation curve for Neural Networks')
    plt.xlabel('alpha value')
    plt.ylabel("Classification score")
    plt.title('classification score vs alpha for Neural Networks')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_ccp_alpha_nn.png'.format(IMAGE_DIR,dataset_name))

