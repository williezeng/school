import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
IMAGE_DIR = 'images/'


def student_drinking_svm(file_name):
    dataset_name = "student_drinking"
    data = pd.read_csv(file_name)
    yy = data["Dalc"]  # only day alcohol data
    xx = data.drop("Dalc", axis=1)  # everything other than day alcohol data
    data_set = train_test_split(xx, yy, test_size=0.3, random_state=3)
    train_X, X_test, train_y, y_test = data_set

    return search_best_params(yy,xx, data_set, dataset_name)

def wine_svm(file_name):
    dataset_name = 'wine'
    data = pd.read_csv(file_name)
    yy = data["Class"]  # only class id data
    xx = data.drop("Class", axis=1)  # everything other than class id data
    data_set = train_test_split(xx, yy, test_size=0.17, random_state=3)

    return search_best_params(yy,xx, data_set, dataset_name)


def mse_training_size(dataset_name, y, x, best_parameters):
    name_to_train_size_map = {'wine': [75, 100, 125, 142, 160, 173],
                              'student_drinking': [230, 250, 280, 300, 316]}
    train_sizes = name_to_train_size_map[dataset_name]

    train_sizes, train_scores, validation_scores = learning_curve(SVC(kernel='linear',C=best_parameters['C'], gamma=best_parameters['gamma']), x, y,
                                                                  train_sizes=train_sizes, scoring='neg_mean_squared_error', cv=5, n_jobs=4)
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Mean Square Error vs Training Size for svm')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_training_size_svm.png'.format(IMAGE_DIR, dataset_name))


def search_best_params(yy,xx, data_set, dataset_name=None):
    train_X, X_test, train_y, y_test = data_set
    name_to_train_size_map = {'wine': [0.000001,0.0001,0.001,0.01,0.1,1],
                              'student_drinking': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    param_grid = [
        {
            'C': name_to_train_size_map[dataset_name],
            'gamma': [0.00001, 0.0001, 0.001]
        }
    ]

    clf = GridSearchCV(SVC(kernel='linear'), param_grid, cv=3)
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
    best_parameters = clf.best_params_
    mse_training_size(dataset_name, yy, xx, best_parameters)
    plot_validation_curves(dataset_name,best_parameters, train_X, train_y)
    return accuracy_score(train_y, train_y_predict), accuracy_score(y_test, y_test_predict), fit_time, test_time


def plot_validation_curves(dataset_name, best_parameters, train_x, train_y):
    # plot C

    name_to_train_size_map = {'wine': [0.000001,0.0001,0.001,0.01,0.1,1],
                              'student_drinking':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    train_scores, test_scores = validation_curve(SVC(kernel='linear', gamma=best_parameters['gamma']), train_x, train_y,
                                                 param_name="C", param_range=name_to_train_size_map[dataset_name], cv=5, n_jobs=5)

    trained = np.mean(train_scores, axis=1)
    tested = np.mean(test_scores, axis=1)
    if dataset_name=='svm':
        import pdb
        pdb.set_trace()
    plt.figure()
    plt.plot(name_to_train_size_map[dataset_name], trained, label='Training score')
    plt.plot(name_to_train_size_map[dataset_name], tested, label='Cross-validation score')
    plt.title('Validation curve for svm')
    plt.xlabel('C value')
    plt.ylabel("Classification score")
    plt.title('classification score vs C value for svm')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_c_svm.png'.format(IMAGE_DIR, dataset_name))
