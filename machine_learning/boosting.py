import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

IMAGE_DIR = 'images/'

def student_drinking(file_name):
    dataset_name = "student_drinking"
    data = pd.read_csv(file_name)
    yy = data["Dalc"]  # only day alcohol data
    xx = data.drop("Dalc", axis=1)  # everything other than day alcohol data
    data_set = train_test_split(xx, yy, test_size=0.25, random_state=3)
    train_X, X_test, train_y, y_test = data_set

    mse_training_size(dataset_name, yy, xx)
    # plot_validation_curves(dataset_name, train_X, train_y)

    return search_best_params(data_set, dataset_name)

def wine(file_name):
    dataset_name = 'wine'
    data = pd.read_csv(file_name)
    yy = data["Class"]  # only class id data
    xx = data.drop("Class", axis=1)  # everything other than class id data
    data_set = train_test_split(xx, yy, test_size=0.3, random_state=3)
    mse_training_size(dataset_name, yy, xx)

    return search_best_params(data_set, dataset_name)

def mse_training_size(dataset_name, y, x):
    dt_stump = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1, ccp_alpha=0.003)
    # clf_boosted = AdaBoostClassifier(base_estimator=dt_stump, random_state=7)
    name_to_train_size_map = {'wine': [1, 25, 50, 75, 100, 125, 142, 160, 173],
                              'student_drinking': [1, 75, 150, 225, 316]}
    train_sizes = name_to_train_size_map[dataset_name]
    train_sizes, train_scores, validation_scores = learning_curve(AdaBoostClassifier(base_estimator=dt_stump, random_state=7), x, y,
                                               train_sizes=train_sizes,
                                               scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Mean Square Error vs Training Size for boosting')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_training_size_b'.format(IMAGE_DIR, dataset_name))


def search_best_params(data_set, dataset_name=None):
    train_X, X_test, train_y, y_test = data_set

    dt_stump = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1, ccp_alpha=0.003)

    param_grid = [
        {
            'base_estimator__max_depth': [1, 2, 3, None],
            'n_estimators': [25, 33, 50, 75, 100],
            'learning_rate': [0.75,0.99]
        }
    ]

    clf = GridSearchCV(AdaBoostClassifier(base_estimator=dt_stump, random_state=7), param_grid, cv=3)
    t0 = time.time()
    clf.fit(train_X, train_y)
    t1 = time.time()
    fit_time = t1 - t0

    train_y_predict = clf.predict(train_X)
    t0 = time.time()
    y_test_predict = clf.predict(X_test)
    t1 = time.time()
    test_time = t1 - t0
    print("Best boosted parameters set found on development set:")
    print(clf.best_params_)
    best_parameters = clf.best_params_
    plot_validation_curves(dataset_name,best_parameters, train_X, train_y)
    return accuracy_score(train_y, train_y_predict), accuracy_score(y_test, y_test_predict), fit_time, test_time

def plot_validation_curves(dataset_name, best_parameters, train_x, train_y):
    # plot Learning rate
    dt_stump = tree.DecisionTreeClassifier(max_depth=best_parameters['base_estimator__max_depth'],
                                           min_samples_leaf=1, ccp_alpha=0.003)
    name_to_train_size_map = {'wine': [0.75,0.88,1],
                              'student_drinking': [0.25,0.33,0.4,0.5,0.75,0.88,1]}
    train_scores, test_scores = validation_curve(AdaBoostClassifier(base_estimator=dt_stump, random_state=7, n_estimators=best_parameters['n_estimators'],), train_x, train_y,
                                                 param_name="learning_rate", param_range=name_to_train_size_map[dataset_name])

    trained = np.mean(train_scores, axis=1)
    tested = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(name_to_train_size_map[dataset_name], trained, label='Training score')
    plt.plot(name_to_train_size_map[dataset_name], tested, label='Cross-validation score')
    plt.title('Validation curve for boosting')
    plt.xlabel('learning_rate value')
    plt.ylabel("Classification score")
    plt.title('classification score vs learning_rate for boosting')
    plt.legend()
    plt.grid()
    plt.savefig('{}{}_learning_rate_b.png'.format(IMAGE_DIR, dataset_name))

