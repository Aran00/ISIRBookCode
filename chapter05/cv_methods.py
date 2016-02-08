__author__ = 'Aran'

from sklearn import cross_validation

def kfold_iterator(n, n_folds, shuffle, random_state):
    kf = cross_validation.KFold(n, n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    if __name__ == '__main__':
        for train, test in kf:
            print("%s %s" % (train, test))
    return kf

def straitified_kfold(labels, n_folds):
    skf = cross_validation.StratifiedKFold(labels, n_folds=n_folds)
    if __name__ == '__main__':
        for train, test in skf:
            print("%s %s" % (train, test))
    return skf

def label_kfold(labels, n_folds):
    lkf = cross_validation.LabelKFold(labels, n_folds=n_folds)
    for train, test in lkf:
        print("%s %s" % (train, test))
    return lkf

def shuffle_kfold(random_state):
    ss = cross_validation.ShuffleSplit(32, n_iter=10, test_size=0.1, random_state=random_state)
    for train_index, test_index in ss:
        print("%s %s" % (train_index, test_index))
    return ss

if __name__ == '__main__':
    kfold_iterator(10, 3, True, 5)
    print '\n'
    # label_kfold([1, 1, 1, 2, 2, 2, 3, 3, 3, 3], 3)
    '''
    #straitified_kfold([0, 0, 1, 1, 2, 2], 3
    shuffle_kfold(0)
    print '\n'
    '''