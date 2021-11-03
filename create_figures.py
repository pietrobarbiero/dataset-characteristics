import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
##################################### USE LATEX ANNOTATION IN MATPLOTLIB LABELS
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#import matplotlib.pyplot as plt
#params = {'text.latex.preamble' : [r'\usepackage{amsfonts}', r'\usepackage{amsmath}']}
#plt.rcParams.update(params)
#####################################
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import sem

# this is a dictionary that translates the notation of the tables into
# the notation used in the paper
label_to_latex = dict()
# standard metrics
label_to_latex["n_samples"] = r"$n$"
label_to_latex["n_features"] = r"$d$"
label_to_latex["n_classes"] = r"$c$"
# Euclidian metrics
label_to_latex["intrinsic_dimensionality"] = r"$\mathfrak{I}$"
label_to_latex["feature_noise"] = r"$\mathfrak{N}$"
label_to_latex["intrinsic_dimensionality_ratio"] = r"$\mathfrak{I}_r$"
label_to_latex["sample_avg_distance"] = r"$\mu_{\mathfrak{D}}$"
label_to_latex["sample_std_distance"] = r"$\sigma_{\mathfrak{D}}$"
# statistical metrics
label_to_latex["levene_pvalue"] = r"$\lambda$"
label_to_latex["levene_stat"] = r"$\lambda_s$"
label_to_latex["levene_success"] = r"$\lambda_r$"
label_to_latex["feature_avg_correlation"] = r"$\rho$"
label_to_latex["feature_avg_kurtosis"] = r"$\kappa$"
label_to_latex["feature_avg_mutual_information"] = r"$\eta$"
label_to_latex["feature_avg_skew"] = r"$\gamma$"
# generalization metrics
label_to_latex["imbalance_ratio_train"] = r"$CI_{train}$"
label_to_latex["imbalance_ratio_val"] = r"$CI_{test}$"
label_to_latex["in_hull_ratio"] = r"$T_{in}$"
label_to_latex["out_hull_ratio"] = r"$T_{out}$"
#label_to_latex["out_hull_ratio"] = r"$T_{out}$" # this does not exist, I think
label_to_latex["imbalance_ratio_in_hull"] = r"$CI_{in}$"
label_to_latex["imbalance_ratio_out_hull"] = r"$CI_{out}$"
label_to_latex["train_f1"] = r"$F1_{train}$"
label_to_latex["val_f1"] = r"$F1_{test}$"
label_to_latex["val_f1_in_hull"] = r"$F1_{in}$"
label_to_latex["val_f1_out_hull"] = r"$F1_{out}$"
label_to_latex["train_accuracy"] = r"$A_{train}$"
label_to_latex["val_accuracy"] = r"$A_{test}$"
label_to_latex["val_accuracy_in_hull"] = r"$A_{in}$"
label_to_latex["val_accuracy_out_hull"] = r"$A_{out}$"
# additional metrics requested by Reviewer 7
label_to_latex["n_features_over_n_samples"] = r"$d/n$"
label_to_latex["intrinsic_dimensionality_over_n_samples"] = r"$\mathfrak{I}/n$"

##############################################################################
# ALL'IMPROVVISO, IL LATEX
matplotlib.rcParams['text.usetex'] = True
params = {'text.latex.preamble' : [r'\usepackage{amsfonts}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

input_file = "data/results-by-fold.csv"
output_folder = "figures/"

if not os.path.exists(output_folder) : os.mkdir(output_folder)
M = pd.read_csv(input_file)

# these lines below just check what happens if we remove lines with missing values
variant = ""
if True :
    # variant for the reviewer; we add the lines, but also remove Levene's stuff, besides stats
    M.drop(["levene_stat", "levene_success"], axis=1, inplace=True)
    # also, reorder columns in the same order as the metrics appearing in the paper
    M = M[[ 
            "data_set_name",
            "model_name",
            "random_state",
            "n_splits",
            "data_set_id",
            "split_idx",
            "n_samples", 
            "n_features", 
            "n_classes", 
            "intrinsic_dimensionality",
            "intrinsic_dimensionality_ratio",
            "feature_noise",
            "sample_avg_distance",
            "sample_std_distance",
            "levene_pvalue",
            "feature_avg_correlation",
            "feature_avg_skew",
            "feature_avg_kurtosis",
            "feature_avg_mutual_information",
            "imbalance_ratio_train",
            "imbalance_ratio_val",
            "in_hull_ratio",
            "out_hull_ratio",
            "imbalance_ratio_in_hull",
            "imbalance_ratio_out_hull",
            "train_accuracy",
            "train_f1",
            "val_accuracy",
            "val_accuracy_in_hull",
            "val_accuracy_out_hull",
            "val_f1",
            "val_f1_in_hull",
            "val_f1_out_hull",
            ]]
    variant = "_revised_for_paper"
if False :
    M.dropna(inplace=True)
    variant = "_no_missing"
if False :
    M.replace("", 1.0, inplace=True)
    M.replace(np.nan, 1.0, inplace=True)
    variant = "_replace_one"
if False :
    M.replace("", float("NaN"), inplace=True)
    columns = [c for c in list(M) if c not in ["data_set_name", "model_name", "random_state", "n_splits", "data_set_id", "split_idx",]]
    for c in columns :
        M[c].fillna(M[c].mean(), inplace=True)
    variant = "_replace_mean"
if False :
    reference_column = dict()
    reference_column["imbalance_ratio_in_hull"] = "imbalance_ratio_out_hull"
    reference_column["imbalance_ratio_out_hull"] = "imbalance_ratio_in_hull"
    reference_column["val_accuracy_in_hull"] = "val_accuracy_out_hull"
    reference_column["val_accuracy_out_hull"] = "val_accuracy_in_hull"
    reference_column["val_f1_in_hull"] = "val_f1_out_hull"
    reference_column["val_f1_out_hull"] = "val_f1_in_hull"
    
    M.replace("", float("NaN"), inplace=True)
    for c in reference_column :
        M[c].fillna(M[reference_column[c]], inplace=True)
        
    variant = "_replace_reference"
    
    
idx_rf = []
idx_lr = []
idx_svc = []
for i, name in enumerate(M["model_name"]):
    if "SVC" in name:
        idx_svc.append(i)
    if "LogisticRegression" in name:
        idx_lr.append(i)
    if "RandomForest" in name:
        idx_rf.append(i)

M_rf = M.iloc[idx_rf]
M_lr = M.iloc[idx_lr]
M_svc = M.iloc[idx_svc]

M_lr["intrinsic_dimensionality"].mean()
M_lr["intrinsic_dimensionality"].sem()
M_lr["feature_avg_correlation"].mean()
M_lr["feature_avg_correlation"].sem()
M_lr["n_samples"].min()
M_lr["n_samples"].max()
M_lr["n_features"].min()
M_lr["n_features"].max()

dataset_stats = {
    "SVC": M_svc,
    "LR": M_lr,
    "RF": M_rf,
}

cmap = sns.color_palette("coolwarm", 50)

# add columns to M
for model, M_s in dataset_stats.items() :
    # in order to place the two columns in the proper position, we first need to find the index for the column "train_accuracy",
    # that is the first one to appear in the final block for quality metrics
    columns = list(M_s)
    index = columns.index("train_accuracy")

    #M_s["n_features_over_n_samples"] = M_s["n_features"] / M_s["n_samples"]
    M_s.insert(loc=index, column="n_features_over_n_samples", value=M_s["n_features"] / M_s["n_samples"])
    #M_s["intrinsic_dimensionality_over_n_samples"] = M_s["intrinsic_dimensionality"] / M_s["n_samples"]
    M_s.insert(loc=index+1, column="intrinsic_dimensionality_over_n_samples", value=M_s["intrinsic_dimensionality"] / M_s["n_samples"])
    print(M_s["n_features_over_n_samples"])
    print(M_s["intrinsic_dimensionality_over_n_samples"])

for dataset, X in dataset_stats.items():
    print(f"{dataset} train F1: {X['train_f1'].mean():.2f} {X['train_f1'].sem():.2e}")
    print(f"{dataset} validation F1: {X['val_f1'].mean():.2f} {X['val_f1'].sem():.2e}")
    print(f"{dataset} validation F1 in hull: {X['val_f1_in_hull'].mean():.2f} {X['val_f1_in_hull'].sem():.2e}")
    print(f"{dataset} validation F1 out hull: {X['val_f1_out_hull'].mean():.2f} {X['val_f1_out_hull'].sem():.2e}")

    # some extra measures, we need to create a new dataframe
    df = pd.DataFrame()
    # abs(F1_train - F1_test)
    df['f1_train_minus_f1_test'] = np.array([ abs(x_1-x_2) for x_1, x_2 in zip(X['train_f1'].values, X['val_f1'].values) ])
    print(f"{dataset} |F1_train-F1_test|: {df['f1_train_minus_f1_test'].mean():2f} ({df['f1_train_minus_f1_test'].mean():.2e}) {df['f1_train_minus_f1_test'].sem():.2e}")

    X_s = X[['train_f1', 'val_f1_in_hull']].dropna()
    f1_train_minus_f1_in = np.array([ abs(x_1-x_2) for x_1, x_2 in zip(X_s['train_f1'].values, X_s['val_f1_in_hull'].values) ])
    print(f"{dataset} |F1_train-F1_in|: {f1_train_minus_f1_in.mean():.2f} ({f1_train_minus_f1_in.mean():.2e}) {sem(f1_train_minus_f1_in):.2e}")

    X_s = X[['train_f1', 'val_f1_out_hull']].dropna()
    f1_train_minus_f1_out = np.array([ abs(x_1-x_2) for x_1, x_2 in zip(X_s['train_f1'].values, X_s['val_f1_out_hull'].values) ])
    print(f"{dataset} |F1_train-F1_out|: {f1_train_minus_f1_out.mean():.2f} ({f1_train_minus_f1_out.mean():.2e}) {sem(f1_train_minus_f1_out):.2e}")
    
    print("%s: Plotting figure of type 1" % dataset)
    matplotlib.rcParams['text.usetex'] = False # deactivate Latex for this plot
    
    plt.figure(figsize=[3, 2])
    # sns.set_style('whitegrid')
    g = sns.jointplot(X['val_f1_out_hull'], X['val_f1_in_hull'])
    plt.title(dataset)
    g.ax_joint.set_xlabel("validation F1 inside the convex hull")
    g.ax_joint.set_ylabel("validation F1 outside the convex hull")
    #plt.show()

    X = X.drop([
        "data_set_name",
        "model_name",
        "random_state",
        "n_splits",
        "data_set_id",
        "split_idx",
    ], axis=1)
    X2 = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    C = X2.corr()
    
    # prepare latex labels and reactivate Latex for the next plot
    matplotlib.rcParams['text.usetex'] = True
    latex_labels = [label_to_latex[c] for c in C.columns.values]
    
    print("%s: Plotting figure of type 2" % dataset)
    plt.figure(figsize=[20,20])
    g = sns.heatmap(C, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar=False, 
                    xticklabels=latex_labels, yticklabels=latex_labels, cmap=cmap, annot_kws={"fontsize":16})
    g.set_xticklabels(labels=latex_labels, fontsize=20, rotation=90)
    g.set_yticklabels(labels=latex_labels, fontsize=20, rotation=0)
    plt.title(dataset)
    plt.savefig(f"{output_folder}/{dataset}_corr{variant}.png", dpi=300)
    plt.savefig(f"{output_folder}/{dataset}_corr{variant}.pdf")
    #plt.show()

    #break

selected = [
    'feature_avg_kurtosis',
    'feature_avg_skew',
    'feature_avg_correlation',
    'intrinsic_dimensionality',
    'intrinsic_dimensionality_ratio',
    'n_features',
    'sample_avg_distance',
    'in_hull_ratio',
    'val_f1',
]

C = M.corr()

C = C.loc[selected][selected]

print("Original labels:", C.columns.values)
latex_labels = [label_to_latex[c] for c in C.columns.values]
print("Latex labels:", latex_labels)

mask = np.zeros_like(C)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=[6, 6])
sns.heatmap(C, vmin=-1, vmax=1, annot=True, cbar=False,
            fmt=".2f", mask=mask, cmap=cmap, square=True,
            xticklabels=latex_labels, yticklabels=latex_labels)
# plt.title(dataset)
plt.tight_layout()
plt.savefig(f"{output_folder}/corr_small{variant}.png", dpi=300)
plt.savefig(f"{output_folder}/corr_small{variant}.pdf")
#plt.show()

import matplotlib as mpl
from matplotlib.colors import ListedColormap
my_cmap = ListedColormap(cmap.as_hex())

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=ax, orientation='horizontal', label='correlation')
plt.tight_layout()
plt.savefig(f"{output_folder}/cbar{variant}.png", dpi=300)
plt.savefig(f"{output_folder}/cbar{variant}.pdf")
#plt.show()

# another correlation heatmap, this time including all variables
print("Another heatmap for the full correlation matrix")
X = M.drop([
    "data_set_name",
    "model_name",
    "random_state",
    "n_splits",
    "data_set_id",
    "split_idx",
], axis=1)
# find index of the "train_f1"
index = list(X).index("train_accuracy")
X.insert(loc=index, column="n_features_over_n_samples", value=X["n_features"] / X["n_samples"])
X.insert(loc=index+1, column="intrinsic_dimensionality_over_n_samples", value=X["intrinsic_dimensionality"] / X["n_samples"])
C = X.corr()
latex_labels = [label_to_latex[c] for c in C.columns.values]
plt.figure(figsize=[20, 20])
g = sns.heatmap(C, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar=False, 
                xticklabels=latex_labels, yticklabels=latex_labels, cmap=cmap, annot_kws={"fontsize":16})
g.set_xticklabels(labels=latex_labels, fontsize=20, rotation=90)
g.set_yticklabels(labels=latex_labels, fontsize=20, rotation=0)
plt.title("Correlation over all ML models")
plt.savefig(f"{output_folder}/corr_large{variant}.png", dpi=300)
plt.savefig(f"{output_folder}/corr_large{variant}.pdf")

# here we will create other figures for Reviewer 7
for model, M_selected in dataset_stats.items() :
    print("Now creating plots for %s, following Reviewer 7's ideas" % model)
    #M_selected = M[M["model_name"].str.contains(model)]
    # create correlation matrix
    C = M_selected.corr()
    # select only a few lines/columns
    rows = ["n_features_over_n_samples", "intrinsic_dimensionality_over_n_samples"]
    columns = [c for c in C.columns.values if c.startswith('train') or c.startswith('val')]
    C_selected = C.loc[rows][columns].values
    # create labels
    latex_labels_x = [label_to_latex[c] for c in columns]
    latex_labels_y = [label_to_latex[r] for r in rows]
    # plot
    figure = plt.figure(figsize=[8,2])
    g = sns.heatmap(C_selected, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar=False, 
                    xticklabels=latex_labels_x, yticklabels=latex_labels_y, cmap=cmap, annot_kws={"fontsize":16})
    plt.title(model)
    plt.savefig("%s/reviewer7_%s%s.png" % (output_folder, model, variant), dpi=300)
    plt.savefig("%s/reviewer7_%s%s.pdf" % (output_folder, model, variant))
    #plt.show()
    plt.close(figure)
    
    # another one
    rows = ["n_features", "n_samples", "intrinsic_dimensionality", "intrinsic_dimensionality_ratio"]
    C_selected = C.loc[rows][columns].values
    latex_labels_x = [label_to_latex[c] for c in columns]
    latex_labels_y = [label_to_latex[r] for r in rows]    
    figure = plt.figure(figsize=[8,4])
    g = sns.heatmap(C_selected, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar=False, 
                    xticklabels=latex_labels_x, yticklabels=latex_labels_y, cmap=cmap, annot_kws={"fontsize":16})
    plt.title(model)
    plt.savefig("%s/reviewer7_%s%s_b.png" % (output_folder, model, variant), dpi=300)
    plt.savefig("%s/reviewer7_%s%s_b.pdf" % (output_folder, model, variant))
    #plt.show()
    plt.close(figure)
    
    # also, save to CSV
    M_selected.to_csv("%s/reviewer7_%s_eureqa.csv" % (output_folder, model), index=False)

    # X.to_csv(f"results/{dataset}_res.csv")

# mu = M.groupby(["data_set_name"]).median()
# sigma = M.groupby(["data_set_name"]).sem()

# M = M.drop([
#     "data_set_id",
#     # "data_set_name",
#     # "model_name",
#     "random_state",
#     "n_splits",
#     "data_set_id",
#     # "split_idx",
# ], axis=1)
# M.to_csv("results/reduced.csv", sep='\t')
