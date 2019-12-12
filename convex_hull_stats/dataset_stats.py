# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero and Alberto Tonda
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
import os
import shutil
from typing import List
import traceback
import math
from joblib import Parallel, delayed
from lazygrid.lazy_estimator import LazyPipeline
from sklearn import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle
import copy
import numpy as np
import pandas as pd
import lazygrid as lg
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")
from .convex_hull_tests import convex_combination_test
from .dataset_measures import dimensionality_stats, homogeneity_class_covariances, \
    feature_correlation_class, normality_departure, information, class_stats
from .db_config import create_chull_stmt, insert_chull_stmt, query_chull_stmt


def convex_hull_stats(model: LazyPipeline, X_train, y_train, X_test, y_test, random_state,
                      n_splits, split_idx, data_set_id, data_set_name):
    db_name = os.path.join(model.database, "database.sqlite")
    scaled_data = False
    test_accuracy_in_hull = -1
    test_accuracy_out_hull = -1
    test_f1_in_hull = -1
    test_f1_out_hull = -1

    # fetch stats from chull_stats table
    query = (data_set_id, random_state, n_splits, split_idx)
    chull_stats = lg.database._load_from_db(db_name, query, create_chull_stmt, query_chull_stmt)
    chull_id = chull_stats[0] if chull_stats is not None else chull_stats

    if chull_id is None:

        # basic properties
        n_samples = X_train.shape[0] + X_test.shape[0]
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        # rescale data
        scaler = StandardScaler()
        ss = scaler.fit(X_train)
        X_train_scaled = ss.transform(X_train)
        X_test_scaled = ss.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled)
        X_test_scaled = pd.DataFrame(X_test_scaled)
        X_train_scaled.index = X_train.index
        X_test_scaled.index = X_test.index
        X_train_scaled.columns = X_train.columns
        X_test_scaled.columns = X_test.columns
        X_train = X_train_scaled
        X_test = X_test_scaled
        scaled_data = True

        # compute training set stats
        levene_stat, levene_pvalue, levene_success = homogeneity_class_covariances(X_train.values, y_train)
        if math.isnan(levene_pvalue):
            levene_pvalue = -1
        if math.isnan(levene_stat):
            levene_stat = -1
        feature_avg_correlation = feature_correlation_class(X_train.values, y_train)
        feature_avg_skew, feature_avg_kurtosis = normality_departure(X_train.values, y_train)
        feature_avg_mutual_information = information(X_train.values, y_train)
        dimensionality, intrinsic_dimensionality, \
        intrinsic_dimensionality_ratio, feature_noise, distances = dimensionality_stats(X_train.values)
        sample_avg_distance = np.average(distances, weights=distances)
        sample_std_distance = np.std(distances)

        # convex hull test
        out_indexes = convex_combination_test(X_train.values, X_test.values)
        in_indexes = [not i for i in out_indexes]
        in_hull_ratio = sum(in_indexes) / y_test.shape[0]
        out_hull_ratio = sum(out_indexes) / y_test.shape[0]
        samples_out_hull_indexes = pickle.dumps(out_indexes, protocol=2)

        # class imbalance
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

        entry = (
            data_set_id, data_set_name, random_state, n_splits, split_idx,
            n_samples, n_features, n_classes,
            intrinsic_dimensionality, intrinsic_dimensionality_ratio, feature_noise,
            sample_avg_distance, sample_std_distance,
            levene_stat, levene_pvalue, levene_success, feature_avg_correlation,
            feature_avg_skew, feature_avg_kurtosis, feature_avg_mutual_information,
            in_hull_ratio, out_hull_ratio, samples_out_hull_indexes,
            imbalance_ratio_train, imbalance_ratio_val, imbalance_ratio_in_hull, imbalance_ratio_out_hull
        )

        chull_stats = lg.database._save_to_db(db_name, entry, query,
                                              create_chull_stmt, insert_chull_stmt,
                                              query_chull_stmt)

        # # chull_id = chull_stats[0]
        # try:
        #     chull_id = chull_stats[0]
        # except TypeError:
        #     print(traceback.format_exc())

    # unpack convex-hull stats
    _, _, _, _, _, _, \
    n_samples, n_features, n_classes, \
    intrinsic_dimensionality, intrinsic_dimensionality_ratio, feature_noise, \
    sample_avg_distance, sample_std_distance, \
    levene_stat, levene_pvalue, levene_success, feature_avg_correlation, \
    feature_avg_skew, feature_avg_kurtosis, feature_avg_mutual_information, \
    in_hull_ratio, out_hull_ratio, out_indexes, \
    imbalance_ratio_train, imbalance_ratio_val, imbalance_ratio_in_hull, imbalance_ratio_out_hull = chull_stats
    out_indexes = pickle.loads(out_indexes)

    # rescale data
    if not scaled_data:
        scaler = StandardScaler()
        ss = scaler.fit(X_train)
        X_train_scaled = ss.transform(X_train)
        X_test_scaled = ss.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled)
        X_test_scaled = pd.DataFrame(X_test_scaled)
        X_train_scaled.index = X_train.index
        X_test_scaled.index = X_test.index
        X_train_scaled.columns = X_train.columns
        X_test_scaled.columns = X_test.columns
        X_train = X_train_scaled
        X_test = X_test_scaled
        scaled_data = True

    # convex hull indexes
    in_indexes = [not i for i in out_indexes]
    X_in_hull = X_test.iloc[in_indexes]
    y_in_hull = y_test[in_indexes]
    X_out_hull = X_test.iloc[out_indexes]
    y_out_hull = y_test[out_indexes]

    # fit model & predict
    model.fit(X_train, y_train)

    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    if len(y_in_hull) > 0:
        y_pred_in_hull = model.predict(X_in_hull)
        test_accuracy_in_hull = accuracy_score(y_in_hull, y_pred_in_hull)
        test_f1_in_hull = f1_score(y_in_hull, y_pred_in_hull, average="weighted")
    if len(y_out_hull) > 0:
        y_pred_out_hull = model.predict(X_out_hull)
        test_accuracy_out_hull = accuracy_score(y_out_hull, y_pred_out_hull)
        test_f1_out_hull = f1_score(y_out_hull, y_pred_out_hull, average="weighted")

    stats = {
        "data_set_id": data_set_id, "data_set_name": data_set_name, "model_name": str(model),

        "random_state": random_state, "n_splits": n_splits, "split_idx": split_idx,
        "n_samples": n_samples, "n_features": n_features, "n_classes": n_classes,

        "intrinsic_dimensionality": intrinsic_dimensionality,
        "intrinsic_dimensionality_ratio": intrinsic_dimensionality_ratio,
        "feature_noise": feature_noise,

        "sample_avg_distance": sample_avg_distance,
        "sample_std_distance": sample_std_distance,

        "levene_stat": levene_stat,
        "levene_pvalue": levene_pvalue,
        "levene_success": levene_success,

        "feature_avg_correlation": feature_avg_correlation,
        "feature_avg_skew": feature_avg_skew,
        "feature_avg_kurtosis": feature_avg_kurtosis,
        "feature_avg_mutual_information": feature_avg_mutual_information,

        "in_hull_ratio": in_hull_ratio,
        "out_hull_ratio": out_hull_ratio,

        "imbalance_ratio_train": imbalance_ratio_train,
        "imbalance_ratio_val": imbalance_ratio_val,
        "imbalance_ratio_in_hull": imbalance_ratio_in_hull,
        "imbalance_ratio_out_hull": imbalance_ratio_out_hull,

        "train_accuracy": train_accuracy,
        "val_accuracy": test_accuracy,
        "val_accuracy_in_hull": test_accuracy_in_hull,
        "val_accuracy_out_hull": test_accuracy_out_hull,

        "train_f1": train_f1, "val_f1": test_f1,
        "val_f1_in_hull": test_f1_in_hull, "val_f1_out_hull": test_f1_out_hull
    }

    stats = pd.DataFrame.from_records([stats]).replace(to_replace=[-1], value=[None])

    return stats


def cross_validation(classifier, X, y, train_index, test_index,
                     random_state, n_splits, split_idx, data_set_id,
                     data_set_name):

    X_train, y_train = X.iloc[train_index], y[train_index]
    X_test, y_test = X.iloc[test_index], y[test_index]

    model = copy.deepcopy(classifier)
    scores = convex_hull_stats(model, X_train, y_train, X_test, y_test, random_state,
                               n_splits, split_idx, data_set_id, data_set_name)

    return scores



def openml_data_set_stats(data_set_id: int, data_set_name: str, classifiers: List = None, db_name: str = None,
                          n_splits: int = 10, random_state: int = 42):
    if not db_name:
        db_name = os.path.join("database", data_set_name)
    if not classifiers:
        classifiers = [
            LazyPipeline([("RandomForestClassifier", RandomForestClassifier())], database=db_name),
            LazyPipeline([("LogisticRegression", LogisticRegression())], database=db_name),
            LazyPipeline([("SVC", SVC())], database=db_name),
        ]

    chull_stats = pd.DataFrame()

    # load data
    X, y, n_classes = lg.datasets.load_openml_dataset(data_id=data_set_id, dataset_name=data_set_name)
    X = pd.DataFrame(X)

    for classifier in classifiers:

        try:
            cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            splits = np.arange(n_splits)

            # We clone the estimator to make sure that all the folds are
            # independent, and that it is pickle-able.
            parallel = Parallel(n_jobs=n_splits, prefer="threads")
            scores = parallel(
                delayed(cross_validation)(
                    copy.deepcopy(classifier), copy.deepcopy(X), y, train_index, test_index, random_state,
                    n_splits, split_idx, data_set_id, data_set_name)
                for (train_index, test_index), split_idx in zip(cv.split(X, y), splits))

            for score in scores:
                chull_stats = pd.concat([chull_stats, score], ignore_index=True)

        except:
            logging.exception(": data set (%d, %s)" % (data_set_id, data_set_name))

    return chull_stats


def openml_stats_all(data_sets: pd.DataFrame = None, classifiers: List = None, db_name: str = None,
                     log_file: str = "./log/data_set_stats.log", output_file: str = "./results/data_set_stats.csv"):

    # initialize logging
    log_dir = os.path.dirname(log_file)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # create output folder
    output_dir = os.path.dirname(output_file)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)


    if data_sets is None:
        data_sets = lg.datasets.fetch_datasets(task="classification", min_classes=2,
                                               max_samples=300, max_features=10)

    # analyze data sets
    results = pd.DataFrame()
    progress_bar = tqdm(np.arange(data_sets.shape[0]), leave=False, position=0)
    for i in progress_bar:
        data_set = data_sets.iloc[i]
        progress_bar.set_description("Analysis of data set: %s" % data_set.name)
        chull_stats = openml_data_set_stats(data_set.did, data_set.name, classifiers, db_name)
        results = pd.concat([results, chull_stats], ignore_index=True)
        results.to_csv(output_file)

    return
