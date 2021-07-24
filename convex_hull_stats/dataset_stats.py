# -*- coding: utf-8 -*-

# Scripts to generate all the data and figures
# Copyright (C) 2020 Pietro Barbiero and Alberto Tonda
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import glob
import os
from typing import List

import joblib
import openml
from openml import OpenMLBenchmarkSuite
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
from .convex_hull_tests import convex_combination_test

def cross_validation(classifier, X, y, train_index, test_index,
                     random_state, n_splits, split_idx, task_id, dataset_name, result_dir):
    logging.info("Starting work on split: %d" % split_idx)

    # extract train and test data
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    model = copy.deepcopy(classifier)

    # data preprocessing
    scaler = StandardScaler()
    ss = scaler.fit(X_train)
    X_train_scaled = ss.transform(X_train)
    X_test_scaled = ss.transform(X_test)

    # directories
    model_name = classifier.steps[-1][1].__class__.__name__
    result_chull_dir = f'{result_dir}/{dataset_name}/{n_splits}-fold-cv/{split_idx}'
    result_model_dir = f'{result_dir}/{dataset_name}/{n_splits}-fold-cv/{split_idx}/{model_name}'
    os.makedirs(result_model_dir, exist_ok=True)

    done_chull_filename = f'{result_chull_dir}/chull.done'
    to_do_chull = not glob.glob(done_chull_filename, recursive=True)

    if to_do_chull:
        # convex hull test
        logging.info("Split: %d - Computing convex hull..." % split_idx)
        out_indexes = convex_combination_test(X_train_scaled, X_test_scaled)
        logging.info("Split: %d - Convex hull computed!" % split_idx)

        logging.info("Split: %d - Saving convex hull results..." % split_idx)
        pd.DataFrame(out_indexes, columns=["index"]).to_csv(f'{result_chull_dir}/out_indexes.csv', index=False)
        (~pd.DataFrame(out_indexes, columns=["index"])).to_csv(f'{result_chull_dir}/in_indexes.csv', index=False)
        pd.DataFrame(train_index, columns=["index"]).to_csv(f'{result_chull_dir}/train_indexes.csv', index=False)
        pd.DataFrame(test_index, columns=["index"]).to_csv(f'{result_chull_dir}/test_indexes.csv', index=False)
        logging.info("Split: %d - Convex hull results saved!" % split_idx)

        open(done_chull_filename, 'w').close()

    done_model_filename = f'{result_model_dir}/stats.done'
    to_do_model = not glob.glob(done_model_filename, recursive=True)

    if to_do_model:
        # fit model & predict
        logging.info("Split: %d - Fitting classifier..." % split_idx)
        model.fit(X_train_scaled, y_train)
        logging.info("Split: %d - Classifier fitted!" % split_idx)

        logging.info("Split: %d - Computing predictions..." % split_idx)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        logging.info("Split: %d - Predictions computed!" % split_idx)

        logging.info("Split: %d - Saving model results..." % split_idx)
        joblib.dump(model, f'{result_model_dir}/{model_name}.joblib')
        pd.DataFrame(y_train, columns=["class_label"]).to_csv(f'{result_chull_dir}/y_train.csv', index=False)
        pd.DataFrame(y_test, columns=["class_label"]).to_csv(f'{result_chull_dir}/y_test.csv', index=False)
        pd.DataFrame(y_train_pred, columns=["class_label"]).to_csv(f'{result_model_dir}/y_train_pred.csv', index=False)
        pd.DataFrame(y_test_pred, columns=["class_label"]).to_csv(f'{result_model_dir}/y_test_pred.csv', index=False)
        logging.info("Split: %d - Model results saved!" % split_idx)

        open(done_model_filename, 'w').close()

    logging.info("Finished work on split: %d" % split_idx)

    return


def compute_dataset_stats(X: np.ndarray, y: np.ndarray, classifiers: List, result_dir: str = "./results",
                          n_splits: int = 10, random_state: int = 42,
                          task_id: int = None, dataset_name: str = None):
    logging.info("%s has %d samples and %d features" % (dataset_name, X.shape[0], X.shape[1]))

    for classifier in classifiers:

        model_name = str([estimator.__class__.__name__ for _, estimator in classifier.steps])
        logging.info("Starting analysis using classifier: %s" % model_name)

        try:
            cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            splits = np.arange(n_splits)

            # slow (for debugging ONLY)
            #for (train_index, test_index), split_idx in zip(cv.split(X, y), splits):
            #    cross_validation(copy.deepcopy(classifier), X, y, train_index, test_index, random_state,
            #                     n_splits, split_idx, task_id, dataset_name, result_dir)

            # We clone the estimator to make sure that all the folds are
            # independent, and that it is pickle-able.
            parallel = joblib.Parallel(n_jobs=n_splits, prefer="threads")
            scores = parallel(
                joblib.delayed(cross_validation)(
                    copy.deepcopy(classifier), X, y, train_index, test_index, random_state,
                    n_splits, split_idx, task_id, dataset_name, result_dir)
                for (train_index, test_index), split_idx in zip(cv.split(X, y), splits))

        except:
            logging.exception(": data set (%d, %s)" % (task_id, dataset_name))

        logging.info("Finished analysis using classifier: %s" % model_name)

    return


def openml_stats_all(benchmark_suite: OpenMLBenchmarkSuite, classifiers: List,
                     n_splits: int, result_dir: str = "./results"):
    # create output folder
    os.makedirs(result_dir, exist_ok=True)

    progress_bar = tqdm(benchmark_suite.tasks, leave=False, position=0)
    for task_id in benchmark_suite.tasks:
        # get data
        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y()
        dataset = task.get_dataset()

        progress_bar.set_description("Analysis of data set: %s" % dataset.name)
        logging.info("Starting analysis of data set: %s" % dataset.name)

        # compute and save stats
        compute_dataset_stats(X, y, classifiers, result_dir, n_splits, task_id=task_id, dataset_name=dataset.name)

        logging.info("Finished analysis of data set: %s" % dataset.name)

    return
