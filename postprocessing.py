"""
Script to perform post-processing of the results.
"""
import os
import numpy as np
import pandas as pd
import re
import sys

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
    
    # this is the root folder with all results
    result_folder = "results/"

    # get list of folders (corresponding to each dataset)
    dataset_folders = [ f.path for f in os.scandir(result_folder) if f.is_dir() ]
    print("Found a total of %d dataset folders!" % len(dataset_folders))

    # filter out folders for which computation is incomplete, using is_experiment_complete
    dataset_folders = [ d for d in dataset_folders if is_experiment_complete(d) ]
    print("Of which, %d correspond to completed experiments." % len(dataset_folders))

    # right now it's not really post-processing, it's just a comparison between the different classifiers
    for dataset_folder in dataset_folders :
        dataset_name = os.path.basename(dataset_folder)
        print("Now analyzing folder for dataset \"%s\"..." % dataset_name)

        # get the list of cross-validation experiments
        cv_folders = [ f.path for f in os.scandir(dataset_folder) if f.is_dir() ]

        for cv_folder in cv_folders :
            print("Now analyzing folder for \"%s\" for dataset \"%s\"..." % (os.path.basename(cv_folder), dataset_name))

            # prepare data structures
            performance = dict()

            # let's start collecting information from each fold
            fold_folders = [ f.path for f in os.scandir(cv_folder) if f.is_dir() ]

            for fold_folder in fold_folders :
                fold_number = int(re.search("([0-9]+)", fold_folder).group(1))

                # read some information, to be used later
                df_test = pd.read_csv(os.path.join(fold_folder, "y_test.csv"))
                y_test = df_test["class_label"].values

                # get the list of folders (here representing different classifiers)
                classifier_folders = [ f.path for f in os.scandir(fold_folder) if f.is_dir() ]
                print("Found %d classifiers for fold %d: \"%s\"" % (len(classifier_folders), fold_number, str(classifier_folders))) 

                for classifier_folder in classifier_folders :
                    # get classifier name
                    classifier_name = os.path.basename(classifier_folder)
                    
                    # get statistics (in this case, accuracy on test)
                    df_pred_test = pd.read_csv(os.path.join(classifier_folder, "y_test_pred.csv"))
                    y_pred_test = df_pred_test["class_label"].values

                    # create local dictionary of metrics, using a list of function pointers!
                    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
                    metrics = [accuracy_score, f1_score, matthews_corrcoef]
                    metrics_dict = { m.__name__ : m(y_test, y_pred_test) for m in metrics }

                    # store performance, as a dictionary (classifier) of dictionaries (metrics) of lists (performance per fold)
                    if classifier_name not in performance : performance[classifier_name] = dict()
                    for metric_name, metric_performance in metrics_dict.items() :
                        if metric_name not in performance[classifier_name] : performance[classifier_name][metric_name] = []
                        performance[classifier_name][metric_name].append(metric_performance) 
            
        # once we are at this point, computation on all cv folders for the dataset is over, so let's draw some conclusions
        for classifier_name, classifier_metrics in performance.items() :
            for metric_name, metric_performance in classifier_metrics.items() :
                print("Classifier \"%s\", metric \"%s\": mean=%.4f; std=%.4f" % (classifier_name, metric_name, np.mean(metric_performance), np.std(metric_performance)))


    return

if __name__ == "__main__" :
    sys.exit( main() )
