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

import linecache
import logging
import numpy as np
import openml
import pandas as pd
import os
import sys
import tracemalloc

# local modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import convex_hull_stats


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def main():
    # tracemalloc.start()

    log_file = "./log/data_set_stats.log"

    # initialize logging
    log_dir = os.path.dirname(log_file)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    
    # instead of fetching all datasets, we get all datasets from OpenML-CC18, but some have missing values; 
    # the pre-processing will be performed in convex_hull_stats.dataset_stats.py 
    #df_datasets = lg.datasets.fetch_datasets(task="classification", min_classes=2, max_features=4000, update_data=True)

    logging.info("Loading benchmark suite OpenML-CC18...")
    benchmark_suite = openml.study.get_suite('OpenML-CC18')
    random_state = 42
    classifiers = dict()

    # let's create pipelines for classifiers that also include hyperparameter tuning
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV

    rf_parameter_grid = {
        'n_estimators': [10, 20, 30, 50, 100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4, 5, 6, 7, 8, None],
        'criterion' :['gini', 'entropy']
        }

    classifiers["RandomForestHT"] = HalvingGridSearchCV(RandomForestClassifier(random_state=random_state), rf_parameter_grid)
    classifiers["RandomForest"] = RandomForestClassifier(random_state=random_state)

    convex_hull_stats.openml_stats_all(benchmark_suite, classifiers, n_splits=10)

    # snapshot = tracemalloc.take_snapshot()
    #
    # sys.stdout = open("profile.txt", "w")
    # display_top(snapshot, limit=100)


if __name__ == "__main__":
    sys.exit(main())
