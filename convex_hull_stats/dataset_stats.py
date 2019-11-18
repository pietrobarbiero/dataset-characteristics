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

from typing import List
import traceback
import math
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import copy
import numpy as np
import pandas as pd
import lazygrid as lg
from logging import Logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from convex_hull_tests import convex_combination_test, generalisation_accuracy, in_hull
from dataset_measures import dimensionality_stats, homogeneity_class_covariances, \
    feature_correlation_class, normality_departure, information, class_stats
from db_config import create_chull_stmt, insert_chull_stmt, query_chull_stmt, \
    create_model_chull_stmt, insert_model_chull_stmt, query_model_chull_stmt


def convex_hull_stats(**kwargs):
    seed = kwargs.get("seed")
    n_splits = kwargs.get("n_splits")
    cv_split = kwargs.get("split_index")
    model: lg.SklearnWrapper = kwargs.get("model")
    random_model = kwargs.get("random_model")
    x_train = kwargs.get("x_train")
    y_train = kwargs.get("y_train")
    x_test = kwargs.get("x_val")
    y_test = kwargs.get("y_val")
    logger: Logger = kwargs.get("logger")

    db_name = model.db_name
    dataset_id = model.dataset_id
    dataset_name = model.dataset_name
    model_name = model.model_name

    # load model
    learner = copy.deepcopy(model)
    learner.set_random_seed(seed, cv_split, random_model)
    learner.load_model()

    # fetch stats from chull_stats table
    query = (dataset_id, seed, n_splits, cv_split)
    chull_stats = lg.database.load_from_db(db_name, query, create_chull_stmt, query_chull_stmt)
    chull_id = chull_stats[0] if chull_stats is not None else chull_stats

    if None not in [learner.model_id, chull_id]:

        if logger: logger.info("\tBoth model and stats found!")

    query_model_chull = (chull_id, learner.model_id)
    accuracy_stats = lg.database.load_from_db(db_name, query_model_chull,
                                              create_model_chull_stmt, query_model_chull_stmt)

    # compute convex-hull stats if they haven't been computed yet
    is_scaled = False
    if chull_id is None:

        if logger: logger.info("\tComputing convex-hull stats...")

        # basic properties
        n_samples = x_train.shape[0] + x_test.shape[0]
        n_features = x_train.shape[1]
        n_classes = len(np.unique(y_train))

        # rescale data
        scaler = StandardScaler()
        ss = scaler.fit(x_train)
        x_train = ss.transform(x_train)
        x_test = ss.transform(x_test)
        is_scaled = True

        # compute training set stats
        levene_stat, levene_pvalue, levene_success = homogeneity_class_covariances(x_train, y_train)
        if math.isnan(levene_pvalue):
            levene_pvalue = -1
        if math.isnan(levene_stat):
            levene_stat = -1
        feature_avg_correlation = feature_correlation_class(x_train, y_train)
        feature_avg_skew, feature_avg_kurtosis = normality_departure(x_train, y_train)
        feature_avg_mutual_information = information(x_train, y_train)
        dimensionality, intrinsic_dimensionality, \
        intrinsic_dimensionality_ratio, feature_noise, distances = dimensionality_stats(x_train)
        sample_avg_distance = np.average(distances, weights=distances)
        sample_std_distance = np.std(distances)

        # convex hull test
        out_indexes = convex_combination_test(x_train, x_test)
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
        imbalance_ratio_test = class_stats(y_test)
        if len(y_in_hull) > 0:
            imbalance_ratio_in_hull = class_stats(y_in_hull)
        if len(y_out_hull) > 0:
            imbalance_ratio_out_hull = class_stats(y_out_hull)

        entry = (
            dataset_id, dataset_name, seed, n_splits, cv_split,
            n_samples, n_features, n_classes,
            intrinsic_dimensionality, intrinsic_dimensionality_ratio, feature_noise,
            sample_avg_distance, sample_std_distance,
            levene_stat, levene_pvalue, levene_success, feature_avg_correlation,
            feature_avg_skew, feature_avg_kurtosis, feature_avg_mutual_information,
            in_hull_ratio, out_hull_ratio, samples_out_hull_indexes,
            imbalance_ratio_train, imbalance_ratio_test, imbalance_ratio_in_hull, imbalance_ratio_out_hull
        )

        chull_stats = lg.database.save_to_db(db_name, entry, query,
                                             create_chull_stmt, insert_chull_stmt,
                                             query_chull_stmt)

        # chull_id = chull_stats[0]
        try:
            chull_id = chull_stats[0]
        except TypeError:
            print(traceback.format_exc())

    # unpack convex-hull stats
    _, _, _, _, _, _, \
    n_samples, n_features, n_classes, \
    intrinsic_dimensionality, intrinsic_dimensionality_ratio, feature_noise, \
    sample_avg_distance, sample_std_distance, \
    levene_stat, levene_pvalue, levene_success, feature_avg_correlation, \
    feature_avg_skew, feature_avg_kurtosis, feature_avg_mutual_information, \
    in_hull_ratio, out_hull_ratio, out_indexes, \
    imbalance_ratio_train, imbalance_ratio_test, imbalance_ratio_in_hull, imbalance_ratio_out_hull = chull_stats

    out_indexes = pickle.loads(out_indexes)

    # train model if it hasn't been fitted yet
    if not accuracy_stats:

        if logger: logger.info("\tTraining model...")

        # rescale data
        if not is_scaled:
            scaler = StandardScaler()
            ss = scaler.fit(x_train)
            x_train = ss.transform(x_train)
            x_test = ss.transform(x_test)

        in_indexes = [not i for i in out_indexes]
        x_in_hull = x_test[in_indexes]
        y_in_hull = y_test[in_indexes]
        x_out_hull = x_test[out_indexes]
        y_out_hull = y_test[out_indexes]

        # fit model & predict
        learner.fit(x_train, y_train)

        y_train_pred = learner.predict(x_train)
        y_test_pred = learner.predict(x_test)
        if len(y_in_hull) > 0:
            y_pred_in_hull = learner.predict(x_in_hull)
        if len(y_out_hull) > 0:
            y_pred_out_hull = learner.predict(x_out_hull)

        # accuracy
        test_accuracy_in_hull = -1
        test_accuracy_out_hull = -1
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        if len(y_in_hull) > 0:
            test_accuracy_in_hull = accuracy_score(y_in_hull, y_pred_in_hull)
        if len(y_out_hull) > 0:
            test_accuracy_out_hull = accuracy_score(y_out_hull, y_pred_out_hull)

        # f1
        test_f1_in_hull = -1
        test_f1_out_hull = -1
        train_f1 = f1_score(y_train, y_train_pred, average="weighted")
        test_f1 = f1_score(y_test, y_test_pred, average="weighted")
        if len(y_in_hull) > 0:
            test_f1_in_hull = f1_score(y_in_hull, y_pred_in_hull, average="weighted")
        if len(y_out_hull) > 0:
            test_f1_out_hull = f1_score(y_out_hull, y_pred_out_hull, average="weighted")

        # save model
        learner.save_model()

        entry = (
            chull_id, learner.model_id, learner.model_name,
            train_accuracy, test_accuracy,
            test_accuracy_in_hull, test_accuracy_out_hull,
            train_f1, test_f1,
            test_f1_in_hull, test_f1_out_hull
        )

        # save stats into database
        query_model_chull = (chull_id, learner.model_id)
        accuracy_stats = lg.database.save_to_db(db_name, entry, query_model_chull,
                                                create_model_chull_stmt, insert_model_chull_stmt,
                                                query_model_chull_stmt)

    # unpack accuracy stats
    _, _, _, _, \
        train_accuracy, test_accuracy, test_accuracy_in_hull, test_accuracy_out_hull, \
        train_f1, test_f1, test_f1_in_hull, test_f1_out_hull = accuracy_stats

    stats = (chull_id, learner.model_id, model_name,
             dataset_id, dataset_name, seed, n_splits, cv_split,
             n_samples, n_features, n_classes,
             intrinsic_dimensionality, intrinsic_dimensionality_ratio, feature_noise,
             sample_avg_distance, sample_std_distance,
             levene_stat, levene_pvalue, levene_success, feature_avg_correlation,
             feature_avg_skew, feature_avg_kurtosis, feature_avg_mutual_information,
             in_hull_ratio, out_hull_ratio,
             train_accuracy, test_accuracy, test_accuracy_in_hull, test_accuracy_out_hull,
             train_f1, test_f1, test_f1_in_hull, test_f1_out_hull,
             imbalance_ratio_train, imbalance_ratio_test, imbalance_ratio_in_hull, imbalance_ratio_out_hull
    )

    return stats


def openml_dataset_stats(dataset_id: int, dataset_name: str,
                         classifiers: List = None, db_name: str = None, logger: Logger = None):

    if not logger:
        logger = lg.initialize_logging(log_name="default-log")
    if not db_name:
        db_name = "default-database"
    if not classifiers:
        classifiers = [RandomForestClassifier(), LogisticRegression(), SVC()]

    # load data
    x, y, n_classes = lg.load_openml_dataset(data_id=dataset_id, dataset_name=dataset_name, logger=logger)

    results = pd.DataFrame()

    for classifier in classifiers:
        model = lg.SklearnWrapper(classifier, dataset_id=dataset_id,
                                  dataset_name=dataset_name, db_name=db_name)

        try:
            score, fitted_models, y_pred_list, y_list = lg.cross_validation(model, x=x, y=y,
                                                                            generic_score=convex_hull_stats,
                                                                            logger=logger)

            score_table = pd.DataFrame.from_dict(score).transpose()

            results = pd.concat([results, score_table])

        except:
            logger.exception(traceback.format_exc())

    if len(results) > 0:
        results.columns = [
            "chull_id", "model_id", "model_name",
            "dataset_id", "dataset_name", "seed", "n_splits", "cv_split",
            "n_samples", "n_features", "n_classes",
            "intrinsic_dimensionality", "intrinsic_dimensionality_ratio", "feature_noise",
            "sample_avg_distance", "sample_std_distance",
            "levene_stat", "levene_pvalue", "levene_success", "feature_avg_correlation",
            "feature_avg_skew", "feature_avg_kurtosis", "feature_avg_mutual_information",
            "in_hull_ratio", "out_hull_ratio",
            "imbalance_ratio_train", "imbalance_ratio_test", "imbalance_ratio_in_hull", "imbalance_ratio_out_hull",
            "train_accuracy", "test_accuracy", "test_accuracy_in_hull", "test_accuracy_out_hull",
            "train_f1", "test_f1", "test_f1_in_hull", "test_f1_out_hull"
        ]

    return results


def openml_stats_all(datasets: pd.DataFrame = None, classifiers: List = None,
                     db_name: str = None, logger: Logger = None):

    if not logger:
        logger = lg.initialize_logging(log_name="default-log")
    if not db_name:
        db_name = "default-database"
    if not classifiers:
        classifiers = [RandomForestClassifier(), LogisticRegression(), SVC()]
    if datasets is None:
        datasets = lg.fetch_datasets(task="classification", min_classes=2,
                                     max_samples=300, max_features=10, logger=logger)

    for index, dataset in datasets.iterrows():
        openml_dataset_stats(dataset.did, dataset.name, classifiers, db_name, logger)

    accuracy_stats = lg.load_all_from_db(db_name=db_name, table_name="MODEL_CHULL")
    chull_stats = lg.load_all_from_db(db_name=db_name, table_name="CHULL_STATS")

    columns = [
        "id", "chull_id", "model_id", "model_name",
        "train_accuracy", "test_accuracy", "test_accuracy_in_hull", "test_accuracy_out_hull",
        "train_f1", "test_f1", "test_f1_in_hull", "test_f1_out_hull"
    ]
    accuracy_stats = pd.DataFrame(accuracy_stats, columns=columns).drop("id", axis=1)
    accuracy_stats = accuracy_stats.replace(-1, None)

    columns = [
        "chull_id",
        "dataset_id", "dataset_name", "seed", "n_splits", "cv_split",
        "n_samples", "n_features", "n_classes",
        "intrinsic_dimensionality", "intrinsic_dimensionality_ratio", "feature_noise",
        "sample_avg_distance", "sample_std_distance",
        "levene_stat", "levene_pvalue", "levene_success", "feature_avg_correlation",
        "feature_avg_skew", "feature_avg_kurtosis", "feature_avg_mutual_information",
        "in_hull_ratio", "out_hull_ratio", "samples_out_hull_indexes",
        "imbalance_ratio_train", "imbalance_ratio_test", "imbalance_ratio_in_hull", "imbalance_ratio_out_hull",
    ]
    chull_stats_single = pd.DataFrame(chull_stats, columns=columns).drop("samples_out_hull_indexes", axis=1)
    chull_stats = pd.DataFrame()
    for classifier in classifiers:
        chull_stats = pd.concat([chull_stats, chull_stats_single])

    results = accuracy_stats.merge(chull_stats, on="chull_id", how="right").drop_duplicates()
    results = results.sort_values(by=["dataset_id", "model_name", "cv_split"])

    return results
