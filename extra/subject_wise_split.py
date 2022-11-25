import numpy as np

def subject_wise_split(x, y, participant, subject_wise=True, split=0.10, seed=42):
    """ Split data into train and test sets via an inter-subject scheme, see:
    Shah, V., Flood, M. W., Grimm, B., & Dixon, P. C. (2022). Generalizability of deep learning models for predicting outdoor irregular walking surfaces. 
    Journal of Biomechanics, 139, 111159. https://doi.org/10.1016/j.jbiomech.2022.111159
    
    Arguments:
        x: nd.array, feature space
        y: nd.array, label class
        participant: nd.array, participant associated with each row in x and y
        subject_wise: bool, choices {True, False}, default = True. True = subject-wise split approach, False random-split
        split: float, number between 0 and 1. Default = 0.10. percentage spilt for test set.
        seed: int. default = 42. Seed selector for numpy random number generator.
    Returns:
        x_train: nd.array, train set for feature space
        x_test: nd.array, test set for feature space
        y_train: nd.array, train set label class
        y_test: nd.array, test set label class
        subject_train: nd.array[string], train set for participants by row of input data
        subjects_test: nd.array[string[, test set for participants by row of input data
    """

    np.random.seed(seed)
    if subject_wise:
        uniq_parti = np.unique(participant)
        num = np.round(uniq_parti.shape[0]*split).astype('int64')
        np.random.shuffle(uniq_parti)
        extract = uniq_parti[0:num]
        test_index = np.array([], dtype='int64')
        for j in extract:
            test_index = np.append(test_index, np.where(participant == j)[0])
        train_index = np.delete(np.arange(len(participant)), test_index)
        np.random.shuffle(test_index)
        np.random.shuffle(train_index)

    else:
        index = np.arange(len(participant)).astype('int64')
        np.random.shuffle(index)
        num = np.round(participant.shape[0] * split).astype('int64')
        test_index = index[0:num]
        train_index = index[num:]

    x_train = x[train_index]
    y_train = y[train_index]
    x_test = x[test_index]
    y_test = y[test_index]
    subject_test = participant[test_index]
    subject_train = participant[train_index]

    return x_train, y_train, x_test, y_test, subject_train, subject_test