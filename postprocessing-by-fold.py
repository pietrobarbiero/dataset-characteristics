"""
Script to perform post-processing of the results. Now cleaned and refactored.

TODO: I need to take into account the structure of the old results CSV; not only, but also compute train accuracy/f1/etc. ...it's probably faster to adopt the same type of naming convention. See local_old_scripts.
"""
import gc # let's try some explicit memory control to avoid crashes
import math
import numpy as np
import openml
import os
import pandas as pd
import re
import sys

from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from convex_hull_stats import homogeneity_class_covariances, feature_correlation_class, normality_departure, \
    information, dimensionality_stats, class_stats


def is_experiment_complete(dataset_folder) :
    """
    Given a folder corresponding to a dataset, this function just checks for the presence of all files indicating completeness of results.
    """
    # get the list of all cross-validation experiments performed (corresponding to folders)
    cv_folders = [ f.path for f in os.scandir(dataset_folder) if f.is_dir() ]

    for cv_folder in cv_folders :
        # get the number of folds that should have been analyzed
        n_folds = int(re.search("([0-9]+)", os.path.basename(cv_folder)).group(1))

        # get the list of all folders, corresponding to each fold
        fold_folders = [ f.path for f in os.scandir(cv_folder) if f.is_dir() ]

        # if there are less folders than the number of hypotethical folds, the experiment is incomplete
        if len(fold_folders) < n_folds : return False

        # this data structure is to take into account all classifiers
        classifiers = dict()

        # otherwise, let's peek inside each fold folder
        for fold_folder in fold_folders :

            # are the computations on the convex hull finished?
            if not os.path.exists(os.path.join(fold_folder, "chull.done")) : return False

            # get the list of folders (corresponding to classifiers)
            classifier_folders = [ f.path for f in os.scandir(fold_folder) if f.is_dir() ]

            for c in classifier_folders :
                if os.path.basename(c) not in classifiers :
                    classifiers[os.path.basename(c)] = True

            # if there are not folders for each of the classifiers previously found, the experiment is not finished
            if len(classifier_folders) < len(classifiers) : return False

            # are the computations for each classifier in each fold finished?
            for classifier_folder in classifier_folders :
                if not os.path.exists(os.path.join(classifier_folder, "stats.done")) : return False

    return True


def main() :

    # TODO restructure everything so that the task_id is used to skip datasets already processed (MUCH faster than loading dataset and checking name)

    # list of datasets in which we have issues, they will be ignored (at the moment, this information is not used)
    datasets_with_issues = [    "mnist_784",        # does not fit in memory during the experiments, probably (?)
                                "Bioresponse",      # does not fit in memory during the experiments, probably (?)
#                                "bank-marketing",   # crashes the analysis, out-of-memory error (now working with gc.collect() after every classifier)
                                "connect-4",        # crashes the analysis, out-of-memory error
                                # "bank-marketing",   # crashes the analysis, out-of-memory error (with Chrome in memory, so it's a tight fit)
                                "Fashion-MNIST", # out-of-memory error
                            ]
    datasets_to_be_ignored = [] + datasets_with_issues

    # read 'results.csv' (if it exists) and check which datasets have already been processed
    output_file = "results-by-fold.csv"
    
    # this is the root folder with all results
    result_folder = "results/"
    print("Preparing statistics dictionary, reading files in the \"%s\" folder..." % result_folder)

    # dictionary that will contain all the final results
    stats = dict()
    stats["data_set_name"] = [] # dataset name
    stats["data_set_id"] = [] # I am not sure we can have this
    stats["model_name"] = [] # classifier
    stats["split_idx"] = [] # id of the fold (0, 1, ...)
    stats["n_samples"] = [] # number of samples
    stats["n_features"] = [] # number of features
    stats["n_classes"] = [] # number of different classes
    stats["n_splits"] = [] # number of splits
    stats["random_state"] = [] # random state
    stats["n_features_over_n_samples"] = []
    stats["intrinsic_dimensionality_over_n_samples"] = []

    # the basic idea is that first we are going to find all the names of the columns in the future dataset result
    # names of the metrics related to the dataset are hard-coded
    dataset_metrics_names = [   'levene_stat', 'levene_pvalue', 'levene_success', 'feature_avg_correlation',
                                'feature_avg_skew', 'feature_avg_kurtosis', 'feature_avg_mutual_information',
                                'dimensionality', 'intrinsic_dimensionality', 'intrinsic_dimensionality_ratio',
                                'feature_noise', 'sample_avg_distance', 'sample_std_distance',
                                'imbalance_ratio_in_hull', 'imbalance_ratio_out_hull', 'imbalance_ratio_train',
                                'imbalance_ratio_val', 'in_hull_ratio', 'out_hull_ratio']

    # add a column to the stats dictionary for each dataset metric
    for metric in dataset_metrics_names : stats[metric] = []

    # now, we go through all folders with the results, each folder is a different experiment on a different dataset; depending on the number
    # of different ML algorithms found, we are going to create the corresponding columns for each metric
    classifier_metrics = ["train_accuracy", "train_f1", "val_accuracy", "val_accuracy_in_hull", "val_accuracy_out_hull", "val_f1", "val_f1_in_hull", "val_f1_out_hull"]
    for metric in classifier_metrics : stats[metric] = []

    # get list of folders (corresponding to each dataset)
    dataset_folders = [ f.path for f in os.scandir(result_folder) if f.is_dir() ]
    print("Found a total of %d dataset folders!" % len(dataset_folders))

    dataset_folders_not_complete = [ d for d in dataset_folders if not is_experiment_complete(d) ]
    print("Incomplete experiments: %d %s" % (len(dataset_folders_not_complete), dataset_folders_not_complete))

    # filter out folders for which computation is incomplete, using is_experiment_complete
    dataset_folders = [ d for d in dataset_folders if is_experiment_complete(d) ]
    print("Complete experiments: %d" % len(dataset_folders))

    # before starting to analyze each experiment, we need to load the benchmark suite to compute some stats 
    dataset_names = [os.path.basename(dataset_folder) for dataset_folder in dataset_folders]

    # TODO this part has to be refactored, now we need to know dataset AND fold
    # check if the output file already exists, read it and take note of the datasets that have already been processed
    if os.path.exists(output_file) :
        print("Found existing output file \"%s\". Reading..." % output_file)
        df = pd.read_csv(output_file)

        # check if the columns of the dataframe are exactly the same as the entries in the dictionary that we are using to collect stats
        if set(df.columns) == set(stats.keys()) :
            
            # get the list of datasets in the file
            datasets_already_treated = list(df["dataset"].unique())
            datasets_treated_but_incomplete = []

            # check if there is an incomplete dataset: for every dataset, for every cv, check if all folds are actually there
            for d in datasets_already_treated :
                df_selected = df[ df["dataset"] == d ]
                for cv in df_selected["cv"].unique() :
                    # parse the 'cv' entry to get the number of expected folds
                    n_folds = int(re.search("([0-9]+)-fold-cv", cv).group(1))
                    if (df_selected[ df_selected["cv"] == cv ].shape[0] < n_folds) :
                        datasets_treated_but_incomplete.append(d)

            print("Found " + str(len(datasets_already_treated)) + " datasets already treated:", datasets_already_treated)
            print("Found " + str(len(datasets_treated_but_incomplete)) + " datasets already treated, but incomplete:", datasets_treated_but_incomplete)
            
            # remove all rows corresponding to incomplete datasets
            df = df[ ~df["dataset"].isin(datasets_treated_but_incomplete) ]
            # finalize the list of datasets to be ignored
            datasets_to_be_ignored += [d for d in datasets_already_treated if d not in datasets_treated_but_incomplete]

            # if everything is ok, convert the dataframe to the current 'stats' dictionary
            stats = df.to_dict(orient='list') # orient='list' creates a dictionary of list
            #print(stats)
        else :
            print("Found unexpected columns in the CSV file, cannot proceed. %d columns in file, %d keys in dictionary." % (len(df.columns), len(stats.keys())))
            dc = sorted(list(df))
            dk = sorted(stats.keys())
            print("Dataset columns:", dc)
            print("Dictionary keys:", dk)

            for item in dk :
                if item not in dc :
                    print("Dictionary key not found in dataset:", item)

    print("Loading benchmark suite \"OpenML-CC18\"...")
    benchmark_suite = openml.study.get_suite('OpenML-CC18')

    # and here we start the folder-by-folder analysis
    for task_id in benchmark_suite.tasks : # [146195,] : # benchmark_suite.tasks :

        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset() # this is really slow, and we would just need the NAME of the dataset, but there is apparently no other way of getting it
        print("\nAnalyzing task %d, on dataset \"%s\"..." % (task_id, dataset.name))

        if dataset.name in dataset_names and dataset.name not in datasets_to_be_ignored :
            dataset_name = dataset.name
            dataset_folder = os.path.join('results', dataset_name)
            print("Starting analysis of dataset \"%s\"..." % dataset_name)

            # get data, impute missing values
            X, y = task.get_X_and_y()

            if np.any(np.isnan(X).flatten()):
                print("Missing samples in task, calling imputer...")
                imputer = KNNImputer()
                X = imputer.fit_transform(X)

            print("Now analyzing folder for dataset \"%s\"..." % dataset_name)

            # get the list of cross-validation experiments
            cv_folders = [ f.path for f in os.scandir(dataset_folder) if f.is_dir() ]

            for cv_folder in cv_folders :
                print("Now analyzing folder for experiment \"%s\" (%s) for dataset \"%s\"..." % (os.path.basename(cv_folder), cv_folders, dataset_name))

                # prepare local data structures
                performance = dict()
                dataset_stats = dict()

                # let's start collecting information from each fold
                fold_folders = [ f.path for f in os.scandir(cv_folder) if f.is_dir() ]

                for fold_folder in fold_folders :

                    fold_number = int(re.search("([0-9]+)", os.path.basename(fold_folder)).group(1))

                    # read some information, to be used later
                    df_train = pd.read_csv(os.path.join(fold_folder, "y_train.csv"))
                    y_train = df_train["class_label"].values
                    df_test = pd.read_csv(os.path.join(fold_folder, "y_test.csv"))
                    y_test = df_test["class_label"].values
                    train_idx = pd.read_csv(os.path.join(fold_folder, "train_indexes.csv")).values.squeeze()
                    test_idx = pd.read_csv(os.path.join(fold_folder, "test_indexes.csv")).values.squeeze()
                    in_indexes = pd.read_csv(os.path.join(fold_folder, "in_indexes.csv")).values.squeeze()
                    out_indexes = pd.read_csv(os.path.join(fold_folder, "out_indexes.csv")).values.squeeze()

                    X_train = X[train_idx]
                    X_test = X[test_idx]

                    # data preprocessing
                    scaler = StandardScaler()
                    ss = scaler.fit(X_train)
                    X_train_scaled = ss.transform(X_train)
                    X_test_scaled = ss.transform(X_test)

                    # let's try to save memory
                    del X_train
                    gc.collect()

                    # compute training set stats, this time with aggressive memory control
                    print("\nSplit: %d - Computing data set stats..." % fold_number)
                    print("\tComputing homogeneity_class_covariances...")
                    levene_stat, levene_pvalue, levene_success = homogeneity_class_covariances(X_train_scaled, y_train)
                    if math.isnan(levene_pvalue):
                        levene_pvalue = -1
                    if math.isnan(levene_stat):
                        levene_stat = -1
                    gc.collect()

                    print("\tComputing feature_correlation_class...")
                    feature_avg_correlation = feature_correlation_class(X_train_scaled, y_train)
                    gc.collect()

                    print("\tComputing normality_departure...")
                    feature_avg_skew, feature_avg_kurtosis = normality_departure(X_train_scaled, y_train)
                    gc.collect()

                    print("\tComputing information...")
                    feature_avg_mutual_information = information(X_train_scaled, y_train)
                    gc.collect()

                    print("\tComputing dimensionality_stats...")
                    dimensionality, intrinsic_dimensionality, intrinsic_dimensionality_ratio, feature_noise, distances = dimensionality_stats(X_train_scaled)
                    gc.collect()

                    sample_avg_distance = np.average(distances, weights=distances)
                    sample_std_distance = np.std(distances)
                    print("Split: %d - Data set stats computed!" % fold_number)

                    # convex hull test
                    print("Split: %d - Computing convex hull..." % fold_number)
                    imbalance_ratio_in_hull = -1
                    imbalance_ratio_out_hull = -1
                    y_in_hull = y_test[in_indexes]
                    y_out_hull = y_test[out_indexes]
                    imbalance_ratio_train = class_stats(y_train)
                    imbalance_ratio_val = class_stats(y_test)
                    if len(y_in_hull) > 0:
                        imbalance_ratio_in_hull = class_stats(y_in_hull)
                    if len(y_out_hull) > 0:
                        imbalance_ratio_out_hull = class_stats(y_out_hull)
                    in_hull_ratio = sum(in_indexes) / y_test.shape[0]
                    out_hull_ratio = sum(out_indexes) / y_test.shape[0]
                    print("Split: %d - Convex hull computed!" % fold_number)

                    # dataset metrics are here reported in the same order as the names hard-coded at the beginning of the main
                    dataset_metrics = [levene_stat, levene_pvalue, levene_success, feature_avg_correlation,
                                            feature_avg_skew, feature_avg_kurtosis, feature_avg_mutual_information,
                                            dimensionality, intrinsic_dimensionality, intrinsic_dimensionality_ratio,
                                            feature_noise, sample_avg_distance, sample_std_distance,
                                            imbalance_ratio_in_hull, imbalance_ratio_out_hull, imbalance_ratio_train,
                                            imbalance_ratio_val, in_hull_ratio, out_hull_ratio]

                    # get the list of folders (here representing different classifiers, acting on the same fold)
                    classifier_folders = [ f.path for f in os.scandir(fold_folder) if f.is_dir() ]
                    print("Found %d classifiers for fold %d: \"%s\"" % (len(classifier_folders), fold_number, str(classifier_folders)))

                    for classifier_folder in classifier_folders :

                        # now, all stats that were previously added at the level of the fold are added here instead,
                        # and copied for each row, of each classifier
                        for dataset_metric_name, dataset_metric in zip(dataset_metrics_names, dataset_metrics):
                            stats[dataset_metric_name].append(dataset_metric)

                        # add the last stat
                        stats["intrinsic_dimensionality_over_n_samples"].append(intrinsic_dimensionality / float(X.shape[0]))

                        stats["data_set_name"].append(dataset_name)
                        stats["data_set_id"].append(task_id)
                        stats["n_splits"].append(cv_folder)
                        stats["split_idx"].append(fold_number)
                        stats["n_samples"].append(X.shape[0])
                        stats["n_features"].append(X.shape[1])
                        stats["n_features_over_n_samples"].append(X.shape[0] / float(X.shape[1]))
                        stats["n_classes"].append(len(np.unique(y)))
                        stats["random_state"].append( 42 ) # in fact, we don't know much about the random state, or do we?

                        # get classifier name
                        classifier_name = os.path.basename(classifier_folder)
                        stats["model_name"].append( classifier_name )

                        # get statistics (in this case, accuracy on test)
                        df_pred_train = pd.read_csv(os.path.join(classifier_folder, "y_train_pred.csv"))
                        y_pred_train = df_pred_train["class_label"].values
                        df_pred_test = pd.read_csv(os.path.join(classifier_folder, "y_test_pred.csv"))
                        y_pred_test = df_pred_test["class_label"].values

                        y_in_hull = y_test[in_indexes]
                        y_out_hull = y_test[out_indexes]
                        y_pred_in_hull = y_pred_test[in_indexes]
                        y_pred_out_hull = y_pred_test[out_indexes]

                        # and now, we compute classifier metrics!
                        # TODO for the moment, we leave MCC to the side
                        print("Split: %d; Classifier: \"%s\" - Computing predictions..." % (fold_number, classifier_name))

                        # training stats
                        stats["train_accuracy"].append( accuracy_score(y_train, y_pred_train) )
                        stats["train_f1"].append( f1_score(y_train, y_pred_train, average='weighted') )
                        
                        # test stats
                        test_accuracy_in_hull, test_f1_in_hull, test_mc_in_hull, test_accuracy_out_hull, test_f1_out_hull, test_mc_out_hull = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                        if len(y_in_hull) > 0:
                            test_accuracy_in_hull = accuracy_score(y_in_hull, y_pred_in_hull)
                            test_f1_in_hull = f1_score(y_in_hull, y_pred_in_hull, average="weighted")
                            test_mc_in_hull = matthews_corrcoef(y_in_hull, y_pred_in_hull)

                        if len(y_out_hull) > 0:
                            test_accuracy_out_hull = accuracy_score(y_out_hull, y_pred_out_hull)
                            test_f1_out_hull = f1_score(y_out_hull, y_pred_out_hull, average="weighted")
                            test_mc_out_hull = matthews_corrcoef(y_out_hull, y_pred_out_hull)

                        stats["val_accuracy"].append( accuracy_score(y_test, y_pred_test) ) 
                        stats["val_accuracy_in_hull"].append( test_accuracy_in_hull )
                        stats["val_accuracy_out_hull"].append( test_accuracy_out_hull )

                        stats["val_f1"].append( f1_score(y_test, y_pred_test, average='weighted') )
                        stats["val_f1_in_hull"].append( test_f1_in_hull )
                        stats["val_f1_out_hull"].append( test_f1_out_hull )

                        print("Split: %d; Classifier: \"%s\" - Predictions computed!" % (fold_number, classifier_name))

                        # call garbage collector to save memory (hopefully)
                        gc.collect()

                    # save partial dictionary, the script crashed with an out-of-memory error,
                    # so it's better to save partial results after every dataset
                    #print(stats)
                    #for k in stats.keys() :
                    #    print("Key \"%s\" has a list with %d elements" % (k, len(stats[k])))
                    df = pd.DataFrame.from_dict(stats)
                    # sort columns by name, EXCEPT 'dataset' that will be placed first
                    sorted_columns = sorted(df.columns)
                    sorted_columns.remove("data_set_name")
                    sorted_columns.remove("data_set_id")
                    sorted_columns.remove("n_splits")
                    sorted_columns.remove("split_idx")
                    sorted_columns = ["data_set_name", "data_set_id", "n_splits", "split_idx"] + sorted_columns
                    df = df.reindex(sorted_columns, axis=1)
                    print("Saving statistics (%d rows x %d columns) to file \"%s\"..." % (df.shape[0], df.shape[1], output_file))
                    df.to_csv(output_file, index=False)

        # end if dataset_name is in the list of folder datasets
        elif dataset.name in datasets_to_be_ignored :
            print("Dataset \"%s\" found in the list of datasets already treated and/or datasets creating issues, skipping..." % dataset.name)

        elif dataset.name not in dataset_names :
            print("No folder found for dataset \"%s\", experiment probably not started (or finished), skipping..." % dataset.name)

    return

if __name__ == "__main__" :
    sys.exit( main() )
