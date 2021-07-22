import boosting
import decision_trees
import knn
import neural_network
import svm
import matplotlib.pyplot as plt

IMAGE_DIR = 'images/'


def main():
    wine_callers('wine.csv')
    student_drinking_callers('student_drinkers.csv')


def wine_callers(data_filename):
    dt_train_acc, dt_test_acc, dt_fit_time, dt_test_time = decision_trees.wine_decision_tree(data_filename)
    print('dt wine training score: {}, testing score: {} with fitted time {} and testing time {}'.format(dt_train_acc, dt_test_acc, dt_fit_time, dt_test_time))

    nn_train_acc, nn_test_acc, nn_fit_time, nn_test_time = neural_network.wine_nn(data_filename)
    print('nn wine training score: {}, testing score: {} with fitted time {} and testing time {}'.format(nn_train_acc, nn_test_acc, nn_fit_time, nn_test_time))

    boosted_train_acc, boosted_test_acc, boosted_fit_time, boosted_test_time = boosting.wine(data_filename)
    print('boosted wine training score: {}, testing score: {} with fitted time {} and testing time {}'.format(boosted_train_acc, boosted_test_acc, boosted_fit_time, boosted_test_time))

    svm_train_acc, svm_test_acc, svm_fit_time, svm_test_time = svm.wine_svm(data_filename)
    print('svm wine training score: {}, testing score: {} with fitted time {} and testing time {}'.format(svm_train_acc,
                                                                                                          svm_test_acc,
                                                                                                          svm_fit_time,
                                                                                                          svm_test_time))
    kn_train_acc, kn_test_acc, kn_fit_time, kn_test_time = knn.wine_knn(data_filename)
    print('knn wine training score: {}, testing score: {} with fitted time {} and testing time {}'.format(kn_train_acc, kn_test_acc, kn_fit_time, kn_test_time))
    plt.close('all')
    plt.figure()
    plt.barh(['DT', 'NN', 'boost', 'SVM', 'KNN'],[dt_test_acc*100, nn_test_acc*100, boosted_test_acc*100, svm_test_acc*100, kn_test_acc*100])
    plt.title('Comparison of testing accuracy for each algorithm')
    plt.xlabel('accuracy % ')
    plt.ylabel("algorithms")
    plt.savefig('{}wine_accuracy.png'.format(IMAGE_DIR))
    plt.close()

    plt.figure()
    plt.barh(['DT', 'NN', 'boost', 'SVM', 'KNN'],[dt_fit_time, nn_fit_time, boosted_fit_time, svm_fit_time, kn_fit_time])
    plt.title('Comparison of fitting time for each algorithm')
    plt.xlabel('timing in seconds')
    plt.ylabel("algorithms")
    plt.savefig('{}wine_fit_time.png'.format(IMAGE_DIR))
    plt.close()

    plt.figure()
    plt.barh(['DT', 'NN', 'boost', 'SVM', 'KNN'],[dt_test_time, nn_test_time, boosted_test_time, svm_test_time, kn_test_time])
    plt.title('Comparison of test time for each algorithm')
    plt.xlabel('timing in seconds')
    plt.ylabel("algorithms")
    plt.savefig('{}wine_test_time.png'.format(IMAGE_DIR))
    plt.close()


def student_drinking_callers(data_filename):
    print('------------------------')
    dt_train_acc, dt_test_acc, dt_fit_time, dt_test_time = decision_trees.student_drinking_decision_tree(data_filename)
    print('dt student_drinking training score: {}, testing score: {} with fitted time {} and testing time {}'.format(dt_train_acc, dt_test_acc, dt_fit_time, dt_test_acc))

    nn_train_acc, nn_test_acc, nn_fit_time, nn_test_time = neural_network.student_drinking_nn(data_filename)
    print('nn student_drinking training score: {}, testing score: {} with fitted time {} and testing time {}'.format(nn_train_acc, nn_test_acc, nn_fit_time, nn_test_time))

    boosted_train_acc, boosted_test_acc, boosted_fit_time, boosted_test_time = boosting.student_drinking(data_filename)
    print('boosted student_drinking training score: {}, testing score: {} with fitted time {} and testing time {}'.format(boosted_train_acc, boosted_test_acc, boosted_fit_time, boosted_test_time))

    svm_train_acc, svm_test_acc, svm_fit_time, svm_test_time = svm.student_drinking_svm(data_filename)
    print('svm student_drinking training score: {}, testing score: {} with fitted time {} and testing time {}'.format(svm_train_acc,
                                                                                                          svm_test_acc,
                                                                                                          svm_fit_time,
                                                                                                          svm_test_time))

    kn_train_acc, kn_test_acc, kn_fit_time, kn_test_time = knn.student_knn(data_filename)
    print('knn student_drinking training score: {}, testing score: {} with fitted time {} and testing time {}'.format(kn_train_acc, kn_test_acc, kn_fit_time, kn_test_time))

    plt.close('all')
    plt.figure()
    plt.barh(['DT', 'NN', 'boost', 'SVM', 'KNN'],[dt_test_acc*100, nn_test_acc*100, boosted_test_acc*100, svm_test_acc*100, kn_test_acc*100])
    plt.title('Comparison of testing accuracy for each algorithm')
    plt.xlabel('accuracy %')
    plt.ylabel("algorithms")
    plt.savefig('{}student_drinking_accuracy.png'.format(IMAGE_DIR))
    plt.close()

    plt.figure()
    plt.barh(['DT', 'NN', 'boost', 'SVM', 'KNN'],[dt_fit_time, nn_fit_time, boosted_fit_time, svm_fit_time, kn_fit_time])
    plt.title('Comparison of fitting time for each algorithm')
    plt.xlabel('timing in seconds')
    plt.ylabel("algorithms")
    plt.savefig('{}student_drinking_fit_time.png'.format(IMAGE_DIR))
    plt.close()

    plt.figure()
    plt.barh(['DT', 'NN', 'boost', 'SVM', 'KNN'],[dt_test_time, nn_test_time, boosted_test_time, svm_test_time, kn_test_time])
    plt.title('Comparison of test time for each algorithm')
    plt.xlabel('timing in seconds')
    plt.ylabel("algorithms")
    plt.savefig('{}student_drinking_test_time.png'.format(IMAGE_DIR))
    plt.close()


if __name__ == '__main__':
    exit(main())
