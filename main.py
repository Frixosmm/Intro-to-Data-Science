import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE as SMT
from imblearn.over_sampling import RandomOverSampler as ROS
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# constants for parameter names
BULK = "bulk"
PERMUTATION = "permutation"
ENSEMBLE = "ensemble"
PREDICT = "predict"


MM_NORMALIZE = "mm_norm"
Z_TRANSFORM = "z_trans"
SMOTE = "smote"
NON_SMOTE = "non_smote"
NO_SUPERSAMPLE = "no_supersample"
PCA_30 = "pca_30"
PCA_56 = "pca_56"
BEST_25 = "best_25"

SEED = 42


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


lower = None
upper = None
def mm_normalize(df, training):
    global lower
    global upper
    if training:
        lower = df.min()
        upper = df.max()

    return (df - lower) / (upper - lower)


mean = None
std = None
def z_transform(df, training):
    global mean, std
    if training:
        mean = df.mean()
        std = df.std()
    return (df - mean) / std


def smote(features, labels):
    sm = SMT(sampling_strategy=1, k_neighbors=5, random_state=SEED)
    return sm.fit_resample(features, labels)


def non_smote(features, labels):
    ros = ROS(sampling_strategy="auto", random_state=SEED)
    return ros.fit_resample(features, labels)


pc = None
def pca(normalized, num, training):
    global pc
    if training:
        pc = PCA(n_components=num, random_state=SEED).fit(normalized)
    return pc.transform(normalized)


fs = None
def best_features(normalized, labels, num, training):
    global fs
    if training:
        fs = SelectKBest(score_func=f_classif, k=num)
        fs.fit(normalized, labels)
    return fs.transform(normalized)


def preprocess(features, labels, normalization, sampling,
               representation, training):
    # training parameter determines wheter the preprocessing steps should be
    # re-trained or used from a previous run

    preprocessed = features.copy()
    lbls = labels.copy()
    # step 1: normalize the data
    if normalization.upper() == MM_NORMALIZE.upper():
        preprocessed = mm_normalize(preprocessed, training)
    elif normalization.upper() == Z_TRANSFORM.upper():
        preprocessed = z_transform(preprocessed, training)
    else:
        raise InputError(normalization,
                         "normalization parameter should be one of: " + MM_NORMALIZE + ", " + Z_TRANSFORM)

    # step 2: sample the data
    if sampling.upper() == SMOTE.upper() and training:
        preprocessed, lbls = smote(preprocessed, lbls)
    elif sampling.upper() == NON_SMOTE.upper() and training:
        preprocessed, lbls = non_smote(preprocessed, lbls)
    elif sampling.upper() == NO_SUPERSAMPLE.upper() or not training:
        pass
    else:
        raise InputError(sampling, "sampling parameter should be one of: " + SMOTE + ", " + NON_SMOTE + ", " + NO_SUPERSAMPLE)

    # step 3: select representations
    if representation.upper() == PCA_30.upper():
        preprocessed = pca(preprocessed, 30, training)
    elif representation.upper() == PCA_56.upper():
        preprocessed = pca(preprocessed, 56, training)
    elif representation.upper() == BEST_25.upper():
        preprocessed = best_features(preprocessed, lbls, 25, training)

    else:
        raise InputError(representation,
                         "representation parameter should be one of: " + PCA_30 + ", " + PCA_56 + ", " + BEST_25)

    return preprocessed, lbls


def calc_dist(point_a, point_b):
    dist = 0
    for index in range(0, len(point_a)):
        dist += (point_a[index] ** 2) - (point_b[index] ** 2)
    return dist


def evaluate(classifier, X, y, metric: str):
    # Determine TPR and FPR so F1 and F10 can be determined
    preds = classifier.predict(X)
    prec_total = 0
    prec_score = 0
    rec_total = 0
    rec_score = 0
    for sample in zip(y, preds):
        if sample[0] == 2:
            rec_total += 1
            if sample[1] == 2:
                rec_score += 1
        if sample[0] == 1:
            prec_total += 1
            if sample[1] == 1:
                prec_score += 1

    if prec_total == 0:
        precision = 0.0
    else:
        precision = prec_score / prec_total

    if rec_total == 0:
        recall = 0.0
    else:
        recall = rec_score / rec_total

    if metric == "precision":
        return precision
    if metric == "recall":
        return recall
    if metric == "F1":
        if recall == 0 or precision == 0:
            return 0
        return 2/(1/recall + 1/precision)
    if metric == "F10":
        if recall == 0 and precision == 0:
            return 0
        return 101 * (precision * recall) / (100 * precision + recall)


def pipeline(training_features, labels,
             normalization, sampling, representation):
    training_features = training_features.values
    labels = labels.values
    for metric in ["precision", "recall", "F1", "F10"]:
        # Placeholders for different parameterizations scores of classifiers
        knns = [[], [], []]
        dt_scores = [[], []]
        rf_scores = []
        kfold = KFold(n_splits=10)
        for train_index, test_index in kfold.split(training_features):
            X_train, y_train = preprocess(training_features[train_index],
                                          labels.ravel()[train_index],
                                          normalization, sampling,
                                          representation, training=True)
            X_test, y_test = preprocess(training_features[test_index],
                                        labels.ravel()[test_index],
                                        normalization, sampling,
                                        representation, training=False)

            for index, k in enumerate([1, 3, 5]):
                knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
                knns[index].append(evaluate(knn, X_test, y_test, metric))

            for index, criterion in enumerate(["gini", "entropy"]):
                dt = DecisionTreeClassifier(criterion=criterion).fit(
                    X_train, y_train)
                dt_scores[index].append(evaluate(dt, X_test, y_test, metric))

            # No parameterizations are explored for RF
            rf = RandomForestClassifier().fit(X_train, y_train)
            rf_scores.append(evaluate(rf, X_test, y_test, metric))

        # Print results as fit for csv
        for index, k in enumerate([1, 3, 5]):
            print(normalization + "," + sampling + "," + representation + "," +
                  "KNN," + str(k) + "," + metric + "," +
                  str(np.mean(knns[index])))
        for index, criterion in enumerate(["gini", "entropy"]):
            print(normalization + "," + sampling + "," + representation + "," +
                  "DT," + criterion + "," + metric + "," +
                  str(np.mean(dt_scores[index])))
        print(normalization + "," + sampling + "," + representation + "," +
              "RF," + "None" + "," + metric + "," + str(np.mean(rf_scores)))


if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) < 2:
        raise InputError(arguments, "please specify the mode to use for the preprocessor: bulk, permutation")

    if arguments[1].upper() == PERMUTATION.upper():
        if len(arguments) < 7:
            raise InputError(arguments, "please give the input features file, input labels file, mode, normalization "
                                        "type, sampling type, representation type")
        if len(arguments) > 7:
            print("WARNING: the following provided arguments will be ignored: " + str(arguments[7:]))
        features_csv = pd.read_csv(arguments[2], header=None)
        labels_csv = pd.read_csv(arguments[3], header=None)
        training_features = features_csv.iloc[:len(labels_csv)]
        unclassified_features = features_csv.iloc[len(labels_csv):]
        pipeline(training_features.copy(),
                 labels_csv.copy(), arguments[4], arguments[5],
                 arguments[6])

    elif arguments[1].upper() == BULK.upper():
        if len(arguments) < 4:
            raise InputError(arguments, "please give the input features file, input labels file, mode")
        if len(arguments) > 4:
            print("WARNING: the following provided arguments will be ignored: "
                  + str(arguments[4:]))

        features_csv = pd.read_csv(arguments[2], header=None)
        labels_csv = pd.read_csv(arguments[3], header=None)
        training_features = features_csv[:len(labels_csv)]
        unclassified_features = features_csv[len(labels_csv):]
        normalizations = [MM_NORMALIZE, Z_TRANSFORM]
        samplings = [SMOTE, NON_SMOTE, NO_SUPERSAMPLE]
        representations = [PCA_30, PCA_56, BEST_25]

        for n in normalizations:
            for s in samplings:
                for r in representations:
                    pipeline(training_features.copy(),
                             labels_csv.copy(), n, s, r)

    elif arguments[1].upper() == ENSEMBLE.upper():
        predict = False
        if len(arguments) < 4:
            raise InputError(arguments, "please give the input features file, input labels file, mode")
        if len(arguments) > 5:
            print("WARNING: the following provided arguments will be ignored: " + str(arguments[5:]))
        elif len(arguments) == 5:
            predict = arguments[4].upper() == PREDICT.upper()
        features_csv = pd.read_csv(arguments[2], header=None)
        labels_csv = pd.read_csv(arguments[3], header=None)

        training_features = features_csv.iloc[:len(labels_csv)]
        unclassified_features = features_csv.iloc[len(labels_csv):]
        labels = labels_csv.copy().values
        training_features = training_features.copy().values

        ensemble_scores = []
        kfold = KFold(n_splits=10)
        eclf = None
        for metric in ["precision", "recall", "F1", "F10"]:
            for train_index, test_index in kfold.split(training_features):
                X_train, y_train = preprocess(training_features[train_index],
                                              labels.ravel()[train_index],
                                              MM_NORMALIZE, SMOTE, BEST_25,
                                              training=True)
                X_test, y_test = preprocess(training_features[test_index],
                                            labels.ravel()[test_index],
                                            MM_NORMALIZE, SMOTE, BEST_25,
                                            training=False)
                dtg = DecisionTreeClassifier(criterion="gini")
                dte = DecisionTreeClassifier(criterion="entropy")
                knn5 = KNeighborsClassifier(n_neighbors=5)
                knn3 = KNeighborsClassifier(n_neighbors=3)
                knn1 = KNeighborsClassifier(n_neighbors=1)
                # Build ensemble from classifiers
                eclf = VotingClassifier(
                    estimators=[('dtg', dtg), ('dte', dte), ('knn5', knn5),
                                ('knn3', knn3), ('knn1', knn1)], voting='hard')
            if not predict:
                eclf.fit(X_train, y_train)
                ensemble_scores.append(evaluate(eclf, X_test, y_test, metric))
                # Don't print things when writing to csv prediction
                print(metric + ": " + str(np.mean(ensemble_scores)))

        # Gather all the samples for the final run to make predictions
        if predict:
            X_train, y_train = preprocess(training_features, labels.ravel(),
                                          MM_NORMALIZE, SMOTE, BEST_25,
                                          training=True)
            eclf.fit(X_train, y_train)
            X_test, _ = preprocess(unclassified_features, labels.ravel(),
                                   MM_NORMALIZE, SMOTE, BEST_25,
                                   training=False)
            predictions = eclf.predict(X_test)
            for index, pred in enumerate(predictions):
                print(str(index) + "," + str(pred))
    else:
        raise InputError(arguments[1], "mode should be one of: permutation, bulk")
