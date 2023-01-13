import numpy as np
import pandas as pd
import sys


def main() :

    df_openmlcc = pd.read_csv("OpenML-CC18.csv")
    openmlcc_datasets = sorted(df_openmlcc["dataset name"].unique())
    
    #df = pd.read_csv("20220322-new-results-by-fold-complete.csv")
    #df = pd.read_csv("20220329-new-results-by-fold-complete.csv")
    #df = pd.read_csv("20220412-new-results-by-fold-complete-final.csv")
    #df = pd.read_csv("20220412-results-by-fold.csv")
    #df = pd.read_csv("20220620-results-by-fold.csv")
    df = pd.read_csv("20220620-results-by-fold-selected-classifiers.csv")

    column_classifier = "model_name"
    column_dataset = "data_set_name"
    column_performance_train = "train_f1"
    column_performance_test = "val_f1"
    column_performance_test_in = "val_f1_in_hull"
    column_performance_test_out = "val_f1_out_hull"

    # first, let's check how many unique classifiers are included
    unique_classifiers = df[column_classifier].unique()
    print("There are %d classifiers: %s" % (len(unique_classifiers), unique_classifiers))

    # then, check how many complete dataset we got for each one
    for classifier in unique_classifiers :

        # select all rows with classifier name
        df_classifier = df[df[column_classifier] == classifier]

        # naive way: just check unique names of datasets
        datasets_completed_by_classifier = df_classifier[column_dataset].unique()
        print("For classifier \"%s\": %d different datasets" % (classifier, len(datasets_completed_by_classifier)))

        if classifier == "MLPClassifierHT" :
            print(sorted(datasets_completed_by_classifier))

        missing_datasets = [ d for d in openmlcc_datasets if d not in datasets_completed_by_classifier ]
        print("%d missing datasets: %s\n" % (len(missing_datasets), str(missing_datasets)))

    # also, some stats
    df_selected = df.select_dtypes(include=[np.float])
    for c in df_selected.columns :

        values = df[c].values
        mean = np.mean(values)
        std = np.std(values)

        print("For column \"%s\", mean=%.4f, std=%.4f" % (c, mean, std))

    # furthermore, get some stats on classifiers' efficacy
    performance_by_dataset = {"dataset": df[column_dataset].unique()}
    global_performance = {"classifier": unique_classifiers, "F1 (train)": [], "F1 (test)": [], "F1 (test in-hull)": [], "F1 (test out-hull)": [],
            "|F1_train-F1_test|": [], "|F1_train-F1_in|": [], "|F1_train-F1_out|": []}
    for classifier in unique_classifiers :

        # select all rows with classifier name
        df_classifier = df[df[column_classifier] == classifier]

        global_performance["F1 (train)"].append("%.4f +/- %.4f" % (np.mean(df_classifier[column_performance_train]), np.std(df_classifier[column_performance_train])))
        global_performance["F1 (test)"].append("%.4f +/- %.4f" % (np.mean(df_classifier[column_performance_test]), np.std(df_classifier[column_performance_test])))
        global_performance["F1 (test in-hull)"].append("%.4f +/- %.4f" % (np.mean(df_classifier[column_performance_test_in]), np.std(df_classifier[column_performance_test_in])))
        global_performance["F1 (test out-hull)"].append("%.4f +/- %.4f" % (np.mean(df_classifier[column_performance_test_out]), np.std(df_classifier[column_performance_test_out])))

        diff_train_test = (df_classifier[column_performance_train] - df_classifier[column_performance_test]).abs()
        global_performance["|F1_train-F1_test|"].append("%.4f +/- %.4f" % (np.mean(diff_train_test), np.std(diff_train_test)))

        diff_train_test_in = (df_classifier[column_performance_train] - df_classifier[column_performance_test_in]).abs()
        global_performance["|F1_train-F1_in|"].append("%.4f +/- %.4f" % (np.mean(diff_train_test_in), np.std(diff_train_test_in)))

        diff_train_test_out = (df_classifier[column_performance_train] - df_classifier[column_performance_test_out]).abs()
        global_performance["|F1_train-F1_out|"].append("%.4f +/- %.4f" % (np.mean(diff_train_test_out), np.std(diff_train_test_out)))

        # go over each dataset, and compute mean + std of that dataset
        for dataset in performance_by_dataset["dataset"] :

            df_dataset = df_classifier[df_classifier[column_dataset] == dataset]

            mean_train_f1 = np.mean(df_dataset[column_performance_train])
            std_train_f1 = np.std(df_dataset[column_performance_train])

            mean_test_f1 = np.mean(df_dataset[column_performance_test])
            std_test_f1 = np.std(df_dataset[column_performance_test])

            mean_test_f1_in = np.mean(df_dataset[column_performance_test_in])
            std_test_f1_in = np.std(df_dataset[column_performance_test_in])

            mean_test_f1_out = np.mean(df_dataset[column_performance_test_out])
            std_test_f1_out = np.std(df_dataset[column_performance_test_out])

            key_train = classifier + " F1 (train)"
            key_test = classifier + " F1 (test)"
            key_test_in = classifier + " F1 (test in-hull)"
            key_test_out = classifier + " F1 (test out-hull)"

            if key_train not in performance_by_dataset :
                performance_by_dataset[key_train] = []

            if key_test not in performance_by_dataset :
                performance_by_dataset[key_test] = []

            if key_test_in not in performance_by_dataset :
                performance_by_dataset[key_test_in] = []

            if key_test_out not in performance_by_dataset :
                performance_by_dataset[key_test_out] = []

            performance_by_dataset[key_train].append("%.4f +/- %.4f" % (mean_train_f1, std_train_f1))
            performance_by_dataset[key_test].append("%.4f +/- %.4f" % (mean_test_f1, std_test_f1))
            performance_by_dataset[key_test_in].append("%.4f +/- %.4f" % (mean_test_f1_in, std_test_f1_in))
            performance_by_dataset[key_test_out].append("%.4f +/- %.4f" % (mean_test_f1_out, std_test_f1_out))

    # save results
    df_performance = pd.DataFrame.from_dict(performance_by_dataset)
    # first, sort by dataset name
    df_performance = df_performance.sort_values(by=["dataset"], key=lambda col : col.str.lower())
    df_performance.to_csv("performance-by-dataset.csv", index=False)

    # also interesting, count number of datasets that have no samples inside the convex hull
    df_no_in = df_performance[ df_performance["RandomForestHT F1 (test in-hull)"] == "nan +/- nan" ]
    print("A total of %d/%d datasets have no test samples inside the convex hull of the training set." % (df_no_in.shape[0], df_performance.shape[0]))

    df_performance = pd.DataFrame.from_dict(global_performance)
    df_performance.to_csv("performance-global.csv", index=False)


    return

if __name__ == "__main__" :
    sys.exit( main() )
