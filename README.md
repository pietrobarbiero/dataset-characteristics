# Generalization in Machine Learning as a Function of Dataset Characteristics #

Scripts to generate all the data and figures used in the paper "Generalization in Machine Learning as a Function of Dataset Characteristics" [[arXiv:2006.15680](https://arxiv.org/abs/2006.15680)]

## Future works

Re-run experiments on the [OpenML-CC18](https://docs.openml.org/benchmark/#list-of-benchmarking-suites) benchmark suite. Perform hyperparameter optimization using GridSearchCV (or similar).

### Statistics to be saved
0. Random seed used for all the experiments
1. Best hyperparameters obtained after hyperparameter optimization
2. For every fold, for every sample, predictions of all classifiers
3. For every fold, stats related to dataset characteristics
4. Predictions on extrapolation/interpolation of other techniques? However, this can be easily re-computed if we save every fold.
