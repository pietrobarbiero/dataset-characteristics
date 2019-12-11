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

create_chull_stmt = '''CREATE TABLE IF NOT EXISTS CHULL_STATS(
        id INTEGER PRIMARY KEY,
        
        dataset_id INTEGER NOT NULL,
        dataset_name TEXT,
        seed INTEGER NOT NULL,
        n_splits INTEGER NOT NULL,
        cv_split INTEGER NOT NULL,
        
        n_samples INTEGER NOT NULL,
        n_features INTEGER NOT NULL,
        n_classes INTEGER NOT NULL,
        
        intrinsic_dimensionality INTEGER NOT NULL,
        intrinsic_dimensionality_ratio REAL NOT NULL,
        feature_noise REAL NOT NULL,
        
        sample_avg_distance REAL NOT NULL,
        sample_std_distance REAL NOT NULL,
        
        levene_stat REAL NOT NULL,
        levene_pvalue REAL NOT NULL,
        levene_success REAL NOT NULL,
        feature_avg_correlation REAL NOT NULL,
        feature_avg_skew REAL NOT NULL,
        feature_avg_kurtosis REAL NOT NULL,
        feature_avg_mutual_information REAL NOT NULL,
        
        in_hull_ratio REAL NOT NULL,
        out_hull_ratio REAL NOT NULL,
        samples_out_hull_indexes BLOB NOT NULL,
        
        imbalance_ratio_train REAL NOT NULL, 
        imbalance_ratio_test REAL NOT NULL, 
        imbalance_ratio_in_hull REAL NOT NULL, 
        imbalance_ratio_out_hull REAL NOT NULL,
        
        UNIQUE (dataset_id, seed, n_splits, cv_split) 
        )'''

insert_chull_stmt = '''INSERT INTO CHULL_STATS(
        dataset_id, dataset_name, seed, n_splits, cv_split,
        n_samples, n_features, n_classes,
        intrinsic_dimensionality, intrinsic_dimensionality_ratio, feature_noise,
        sample_avg_distance, sample_std_distance, 
        levene_stat, levene_pvalue, levene_success, feature_avg_correlation, 
        feature_avg_skew, feature_avg_kurtosis, feature_avg_mutual_information,
        in_hull_ratio, out_hull_ratio, samples_out_hull_indexes,
        imbalance_ratio_train, imbalance_ratio_test, imbalance_ratio_in_hull, imbalance_ratio_out_hull)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

query_chull_stmt = '''SELECT * FROM CHULL_STATS
                      WHERE dataset_id=? AND seed=? AND n_splits=? AND cv_split=?'''

# create_model_chull_stmt = '''CREATE TABLE IF NOT EXISTS MODEL_CHULL(
#         id INTEGER PRIMARY KEY,
#
#         chull_id INTEGER NOT NULL,
#         model_id INTEGER NOT NULL,
#
#         model_name TEXT NOT NULL,
#
#         train_accuracy REAL NOT NULL,
#         test_accuracy REAL NOT NULL,
#         test_accuracy_in_hull REAL NOT NULL,
#         test_accuracy_out_hull REAL NOT NULL,
#
#         train_f1 REAL NOT NULL,
#         test_f1 REAL NOT NULL,
#         test_f1_in_hull REAL NOT NULL,
#         test_f1_out_hull REAL NOT NULL,
#
#         UNIQUE (chull_id, model_id),
#         FOREIGN KEY (model_id) REFERENCES MODEL(id)
#         FOREIGN KEY (chull_id) REFERENCES CHULL_STATS(id)
#         )'''
#
# insert_model_chull_stmt = '''INSERT INTO MODEL_CHULL(
#         chull_id, model_id, model_name,
#         train_accuracy, test_accuracy, test_accuracy_in_hull, test_accuracy_out_hull,
#         train_f1, test_f1, test_f1_in_hull, test_f1_out_hull)
#         VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
#
# query_model_chull_stmt = '''SELECT * FROM MODEL_CHULL
#                             WHERE chull_id=? AND model_id=?'''
