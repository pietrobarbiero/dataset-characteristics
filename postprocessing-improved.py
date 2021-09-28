"""
Script to perform post-processing of the results. Now cleaned and refactored.
"""
import math
import os
import numpy as np
import openml
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

    # list of datasets in which we have issues, they will be ignored (at the moment, this information is not used)
    datasets_with_issues = ["mnist_784", "Bioresponse"]

    # TODO read 'results.csv' (if it exists) and check which datasets have already been processed
    output_file = "results.csv"
    
    # this is the root folder with all results
    result_folder = "results/"
    print("Preparing statistics dictionary, reading files in the \"%s\" folder..." % result_folder)

    # dictionary that will contain all the final results
    stats = dict()
    stats["dataset"] = [] # dataset name
    stats["cv"] = [] # type of cross-validation (10-fold, 5-fold, ...)

    # the basic idea is that first we are going to find all the names of the columns in the future dataset result
    # names of the metrics related to the dataset are hard-coded
    dataset_metrics_names = [   'levene_stat', 'levene_pvalue', 'levene_success', 'feature_avg_correlation',
                                'feature_avg_skew', 'feature_avg_kurtosis', 'feature_avg_mutual_information',
                                'dimensionality', 'intrinsic_dimensionality', 'intrinsic_dimensionality_ratio',
                                'feature_noise', 'sample_avg_distance', 'sample_std_distance',
                                'imbalance_ratio_in_hull', 'imbalance_ratio_out_hull', 'imbalance_ratio_train',
                                'imbalance_ratio_val', 'in_hull_ratio', 'out_hull_ratio']

    # add a column to the stats dictionary for each dataset metric, considering mean and std
    for metric in dataset_metrics_names :
        stats[metric + " (mean)"] = []
        stats[metric + " (std)"] = []

    # now, we go through all folders with the results, each folder is a different experiment on a different dataset; depending on the number
    # of different ML algorithms found, we are going to create the corresponding columns for each metric
    classifier_metrics_names = [
        accuracy_score.__name__,
        matthews_corrcoef.__name__,
        f1_score.__name__,
            ]

    # get list of folders (corresponding to each dataset)
    dataset_folders = [ f.path for f in os.scandir(result_folder) if f.is_dir() ]
    print("Found a total of %d dataset folders!" % len(dataset_folders))

    dataset_folders_not_complete = [ d for d in dataset_folders if not is_experiment_complete(d) ]
    print("Incomplete experiments: %d %s" % (len(dataset_folders_not_complete), dataset_folders_not_complete))

    # filter out folders for which computation is incomplete, using is_experiment_complete
    dataset_folders = [ d for d in dataset_folders if is_experiment_complete(d) ]
    print("Complete experiments: %d" % len(dataset_folders))

    # let's get a list of all ML algorithms used among all folders; they correspond to the sub-folder names
    ml_algorithm_names = []
    for d in dataset_folders : # for each dataset folder
        for v in [ f.path for f in os.scandir(d) if f.is_dir() ] : # for each N-fold cross-validation
            for fold in [ f.path for f in os.scandir(v) if f.is_dir() ] : # for each fold
                ml_algorithm_names.extend( [ f.name for f in os.scandir(fold) if f.is_dir() ] )
    ml_algorithm_names = sorted(list(set(ml_algorithm_names)))
    print("Here is the list of all ML algorithms appearing at least once in the folders:", ml_algorithm_names)

    # add entries to the dictionary for each combination of classifier metric and classifier name
    for metric_name in classifier_metrics_names :
        for classifier_name in ml_algorithm_names :
            for hull in ["", "_in_hull", "_out_hull"] : # three possibilities: regular metric, metric in-hull, metric out-hull
                stats[metric_name + hull + " " + classifier_name + " (mean)"] = []
                stats[metric_name + hull + " " + classifier_name + " (std)"] = []

    # before starting to analyze each experiment, we need to load the benchmark suite to compute some stats 
    dataset_names = [os.path.basename(dataset_folder) for dataset_folder in dataset_folders]

    print("Loading benchmark suite \"OpenML-CC18\"...")
    benchmark_suite = openml.study.get_suite('OpenML-CC18')

    # and here we start the folder-by-folder analysis
    for task_id in benchmark_suite.tasks:

        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        print("Analyzing task %d, on dataset \"%s\"..." % (task_id, dataset.name))

        if dataset.name in dataset_names :
            dataset_name = dataset.name
            dataset_folder = os.path.join('results', dataset_name)
            print("\nStarting analysis of dataset \"%s\"..." % dataset_name)

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

                # add dataset name to the dictionary, with cv folder name
                stats["dataset"].append(dataset_name)
                stats["cv"].append(cv_folder)

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

                    # compute training set stats
                    print("\nSplit: %d - Computing data set stats..." % fold_number)
                    levene_stat, levene_pvalue, levene_success = homogeneity_class_covariances(X_train_scaled, y_train)
                    if math.isnan(levene_pvalue):
                        levene_pvalue = -1
                    if math.isnan(levene_stat):
                        levene_stat = -1
                    feature_avg_correlation = feature_correlation_class(X_train_scaled, y_train)
                    feature_avg_skew, feature_avg_kurtosis = normality_departure(X_train_scaled, y_train)
                    feature_avg_mutual_information = information(X_train_scaled, y_train)
                    dimensionality, intrinsic_dimensionality, intrinsic_dimensionality_ratio, feature_noise, distances = dimensionality_stats(X_train_scaled)
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

                    if dataset_name not in dataset_stats: dataset_stats[dataset_name] = dict()
                    for dataset_metric_name, dataset_metric in zip(dataset_metrics_names, dataset_metrics):
                        if dataset_metric_name not in dataset_stats[dataset_name]: dataset_stats[dataset_name][dataset_metric_name] = []
                        dataset_stats[dataset_name][dataset_metric_name].append(dataset_metric)

                    # get the list of folders (here representing different classifiers, acting on the same fold)
                    classifier_folders = [ f.path for f in os.scandir(fold_folder) if f.is_dir() ]
                    print("Found %d classifiers for fold %d: \"%s\"" % (len(classifier_folders), fold_number, str(classifier_folders)))

                    for classifier_folder in classifier_folders :
                        # get classifier name
                        classifier_name = os.path.basename(classifier_folder)

                        # get statistics (in this case, accuracy on test)
                        df_pred_train = pd.read_csv(os.path.join(classifier_folder, "y_train_pred.csv"))
                        y_pred_train = df_pred_train["class_label"].values
                        df_pred_test = pd.read_csv(os.path.join(classifier_folder, "y_test_pred.csv"))
                        y_pred_test = df_pred_test["class_label"].values

                        y_in_hull = y_test[in_indexes]
                        y_out_hull = y_test[out_indexes]
                        y_pred_in_hull = y_pred_test[in_indexes]
                        y_pred_out_hull = y_pred_test[out_indexes]

                        # scores
                        # create local dictionary of metrics, using a list of function pointers!
                        # update: it works poorly, because some metrics need special values for their arguments in specific cases
                        print("Split: %d; Classifier: \"%s\" - Computing predictions..." % (fold_number, classifier_name))
                        test_accuracy_in_hull, test_f1_in_hull, test_mc_in_hull, test_accuracy_out_hull, test_f1_out_hull, test_mc_out_hull = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                        if len(y_in_hull) > 0:
                            test_accuracy_in_hull = accuracy_score(y_in_hull, y_pred_in_hull)
                            test_f1_in_hull = f1_score(y_in_hull, y_pred_in_hull, average="weighted")
                            test_mc_in_hull = matthews_corrcoef(y_in_hull, y_pred_in_hull)

                        if len(y_out_hull) > 0:
                            test_accuracy_out_hull = accuracy_score(y_out_hull, y_pred_out_hull)
                            test_f1_out_hull = f1_score(y_out_hull, y_pred_out_hull, average="weighted")
                            test_mc_out_hull = matthews_corrcoef(y_out_hull, y_pred_out_hull)

                        metrics_dict = dict()
                        metrics_dict[accuracy_score.__name__] = accuracy_score(y_test, y_pred_test)
                        metrics_dict[matthews_corrcoef.__name__] = matthews_corrcoef(y_test, y_pred_test)

                        if len(np.unique(y_test)) == 2 :
                            metrics_dict[f1_score.__name__] = f1_score(y_test, y_pred_test)
                        else :
                            metrics_dict[f1_score.__name__] = f1_score(y_test, y_pred_test, average='weighted')
                        metrics_dict[accuracy_score.__name__ + '_in_hull'] = test_accuracy_in_hull
                        metrics_dict[f1_score.__name__ + '_in_hull'] = test_f1_in_hull
                        metrics_dict[matthews_corrcoef.__name__ + '_in_hull'] = test_mc_in_hull

                        metrics_dict[accuracy_score.__name__ + '_out_hull'] = test_accuracy_out_hull
                        metrics_dict[f1_score.__name__ + '_out_hull'] = test_f1_out_hull
                        metrics_dict[matthews_corrcoef.__name__ + '_out_hull'] = test_mc_out_hull
                        print("Split: %d; Classifier: \"%s\" - Predictions computed!" % (fold_number, classifier_name))

                        # store performance, as a dictionary (classifier) of dictionaries (metrics) of lists (performance per fold)
                        if classifier_name not in performance : performance[classifier_name] = dict()
                        for metric_name, metric_performance in metrics_dict.items() :
                            if metric_name not in performance[classifier_name] : performance[classifier_name][metric_name] = []
                            performance[classifier_name][metric_name].append(metric_performance)

            # once we are at this point, computation on all folds for the experiment is over, so let's draw some conclusions
            keys_found = []
            for classifier_name, classifier_metrics in performance.items() :
                for metric_name, metric_performance in classifier_metrics.items() :
                    c_mean = np.mean(metric_performance)
                    c_std = np.std(metric_performance)
                    print("Classifier \"%s\", metric \"%s\": mean=%.4f; std=%.4f" % (classifier_name, metric_name, c_mean, c_std))

                    # and save everything to the dictionary structure, to be later converted to dataframe
                    key_name_mean = metric_name + " " + classifier_name + " (mean)"
                    key_name_std = metric_name + " " + classifier_name + " (std)"
                    stats[key_name_mean].append(c_mean)
                    stats[key_name_std].append(c_std)

                    # record that we had stats for this particular combination of metric and classifier
                    keys_found.append(metric_name + " " + classifier_name)

            # once we are at this point, computation on all cv folders for the dataset is over, so let's draw some conclusions
            for dataset_name, dataset_metrics in dataset_stats.items() :
                for metric_name, metric_performance in dataset_metrics.items() :
                    c_mean = np.mean(metric_performance)
                    c_std = np.std(metric_performance)
                    print("Dataset \"%s\", metric \"%s\": mean=%.4f; std=%.4f" % (dataset_name, metric_name, c_mean, c_std))

                    # and save everything to the dictionary structure, to be later converted to dataframe
                    key_name_mean = metric_name + " (mean)"
                    key_name_std = metric_name + " (std)"
                    stats[key_name_mean].append(c_mean)
                    stats[key_name_std].append(c_std)

                    # record that we had stats for this particular combination of metric and classifier
                    keys_found.append(metric_name)

            # TODO this could be made much easier, just go through all entries in the dictionary and enlarge all lists that are not the largest list
            # here, we must check whether there are all results for all metrics and all classifiers; if that is
            # not the case, we add other 'None' values for everything in the lists
            for classifier_name in ml_algorithm_names :
                for metric_name in classifier_metrics_names :
                    if metric_name + " " + classifier_name not in keys_found :
                        key_name_mean = metric_name + " " + classifier_name + " (mean)"
                        key_name_std = metric_name + " " + classifier_name + " (std)"
                        stats[key_name_mean].append(None)
                        stats[key_name_std].append(None)

            # update: save partial dictionary, the script crashed with an out-of-memory error,
            # so it's better to save partial results after every dataset

            # sanitize dictionary: if some of the lists are shorter than the longest, remove them
            # this can happen when some of the experiments are not over for some of the classifiers
            print(stats)
            longest_list_size = 0
            for key, key_list in stats.items() :
                if len(key_list) > longest_list_size :
                    longest_list_size = len(key_list)

            keys_to_be_removed = []
            for key, key_list in stats.items() :
                if len(key_list) < longest_list_size :
                    keys_to_be_removed.append(key)

            for key in keys_to_be_removed : del stats[key]

            print("Saving statistics to file \"%s\"..." % output_file)
            df = pd.DataFrame.from_dict(stats)
            # sort columns by name, EXCEPT 'dataset' that will be placed first
            sorted_columns = sorted(df.columns)
            print(sorted_columns)
            sorted_columns.remove("dataset")
            sorted_columns = ["dataset"] + sorted_columns
            df = df.reindex(sorted_columns, axis=1)
            df.to_csv(output_file, index=False)

        # end if dataset_name is in the list of folder datasets

    return

if __name__ == "__main__" :
    sys.exit( main() )
