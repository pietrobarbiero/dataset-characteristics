import pandas as pd
import seaborn as sns
import sys

# this is to use Latex in the labels
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
params = {'text.latex.preamble' : [r'\usepackage{amsfonts}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

"""
Get some statistics from the CSV file.
"""
def get_statistics(df) :

    statistics_string = ""

    # how many datasets?
    datasets = df["dataset"].unique()
    statistics_string += "The CSV file contains %d different datasets: " % len(datasets)
    statistics_string += str(datasets)
    statistics_string += "\n"

    # how many classifiers? get column names, see how many start with "accuracy_score " <-- there is a \s
    classifiers = [c.split(" ")[1] for c in df.columns if c.startswith("accuracy_score ")]
    statistics_string += "Found %d different classifiers:" % len(classifiers)
    statistics_string += str(classifiers)
    statistics_string += "\n"

    # for how many classifiers do we have complete statistics? check accuracy_score for NaNs
    for c in classifiers :
        missing_values = df["accuracy_score " + c].isna().sum()
        statistics_string += "Classifier %s has %d missing fold values\n" % (c, missing_values)

    return statistics_string

"""
Main.
"""
def main() :

    # default CSV with information, maybe it can be overwritten by command line
    file_data = "data/results-by-fold.csv"
    if len(sys.argv) > 1 : file_data = sys.argv[1]

    print("Loading data from \"%s\"..." % file_data)
    df = pd.read_csv(file_data)
    print(df)

    # print out some statistics, just to confirm that everything is in order
    print(get_statistics(df))

    # right now, we have three classifiers: LogisticRegression, RandomForest, SVC; each one with and without hyperparameter tuning
    # at the moment, SVC's results are still incomplete, so let's focus on the other two
    classifiers = ["LogisticRegressionHT", "RandomForestHT"] #, "SVC"]

    # metrics that are computed for each classifier
    metrics = ["accuracy_score", "accuracy_score_in_hull", "accuracy_score_out_hull", "f1_score", "f1_score_in_hull", "f1_score_out_hull"]

    # this is the order in which the different columns appear in the paper; also their Latex code
    columns_to_latex = {
            "dataset" : "",
            "model_name" : "",
            "random_state" : "",
            "n_splits" : "",
            "data_set_id" : "",
            "split_idx" : "",
            "n_samples" : r"$n$", 
            "n_features" : r"$d$", 
            "n_classes" : r"$c$", 
            "intrinsic_dimensionality" : r"$\mathfrak{I}$",
            "intrinsic_dimensionality_ratio" : r"$\mathfrak{I}_r$",
            "feature_noise" : r"$\mathfrak{N}$",
            "sample_avg_distance" : r"$\mu_{\mathfrak{D}}$",
            "sample_std_distance" : r"$\sigma_{\mathfrak{D}}$",
            "levene_pvalue" : r"$\lambda$",
            "feature_avg_correlation" : r"$\rho$",
            "feature_avg_skew" : r"$\gamma$",
            "feature_avg_kurtosis" : r"$\kappa$",
            "feature_avg_mutual_information" : r"$\eta$",
            "imbalance_ratio_train" : r"$CI_{train}$",
            "imbalance_ratio_val" : r"$CI_{test}$",
            "in_hull_ratio" : r"$T_{in}$",
            "out_hull_ratio" : r"$T_{out}$",
            "imbalance_ratio_in_hull" : r"$CI_{in}$",
            "imbalance_ratio_out_hull" : r"$CI_{out}$",
            "train_accuracy" : r"$A_{train}$",
            "train_f1" : r"$F1_{train}$",
            "val_accuracy" : r"$A_{test}$",
            "val_accuracy_in_hull" : r"$A_{in}$",
            "val_accuracy_out_hull" : r"$A_{out}$",
            "val_f1" : r"$F1_{test}$",
            "val_f1_in_hull" : r"$F1_{in}$",
            "val_f1_out_hull" : r"$F1_{out}$",
            "n_features_over_n_samples" : r"$d/n$",
            "intrinsic_dimensionality_over_n_samples" : r"$\mathfrak{I}/n$",
            }

    # now, there are N types of matrices we are interested in:
    # - one contains all statistics, and an aggregation of all the data for all classifiers
    # - others contain all statistics, and data for each considered classifier

    # let's start from the BIG ONE
    # here are the columns

    # classifier by classifier
    for classifier in classifiers :
        
        print("Now working on classifier \"%s\"..." % classifier)

        # remove all columns that refer to a metric, but do not mention the classifier
        columns_to_be_removed = []
        for c in df.columns :
            is_metric_column = False
            for m in metrics :
                if c.find(m) != -1 :
                    is_metric_column = True

            if is_metric_column and c.find(classifier) == -1 :
                columns_to_be_removed.append(c)

        print("These columns will not be considered:", columns_to_be_removed)

        # drop the columns
        df_classifier = df.drop(columns_to_be_removed, axis=1)
        # also drop columns that are not numeric
        df_classifier = df_classifier.drop(["dataset", "cv", "fold"], axis=1)

        # now, rename all columns that mention the classifier
        columns_renaming = {}
        for c in df_classifier.columns :
            if c.find(classifier) != -1 :
                tokens = c.split(" ")
                columns_renaming[c] = tokens[0]

        df_classifier.rename(columns=columns_renaming, inplace=True)

        # finally compute correlation matrix
        C = df_classifier.corr()
        latex_labels = [columns_to_latex[c] for c in C.columns.values]
        plt.figure(figsize=[20,20])
        g = sns.heatmap(C, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar=False,
                    xticklabels=latex_labels, yticklabels=latex_labels, cmap=cmap, annot_kws={"fontsize":16})
        g.set_xticklabels(labels=latex_labels, fontsize=20, rotation=90)
        g.set_yticklabels(labels=latex_labels, fontsize=20, rotation=0)
        plt.title(dataset)
        plt.savefig(f"results/corr.png", dpi=300)
        plt.savefig(f"results/corr.pdf")

    return

if __name__ == "__main__" :
    sys.exit( main() )
