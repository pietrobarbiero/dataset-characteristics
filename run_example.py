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

import lazygrid as lg
import linecache
import logging
import numpy as np
import openml
import pandas as pd
import os
import sys
import tracemalloc

# local modules
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
    # the pre-processing is performed in convex_hull_stats.dataset_stats.py 
    #df_datasets = lg.datasets.fetch_datasets(task="classification", min_classes=2, max_features=4000, update_data=True)

    logging.info("Loading benchmark suite OpenML-CC18...")
    benchmark_suite = openml.study.get_suite('OpenML-CC18')

    # unfortunately, the datasets have to be arranged in a pd.Dataframe,
    # otherwise the rest of the code starts crying. It's Pietro's fault.
    # blame him, not me
    dataset_dictionary = {
            "name" : [],
            "did" : [],
            "n_samples" : [],
            "n_features" : [],
            "n_classes" : []
            }
    
    for task_id in benchmark_suite.tasks :
        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y()
        dataset = task.get_dataset()

        dataset_dictionary["name"].append(dataset.name)
        dataset_dictionary["did"].append(dataset.dataset_id)
        dataset_dictionary["n_samples"].append(X.shape[0])
        dataset_dictionary["n_features"].append(X.shape[1])
        dataset_dictionary["n_classes"].append(len(np.unique(y)))

    # create dataframe
    print(dataset_dictionary)
    df_datasets = pd.DataFrame.from_dict(dataset_dictionary)
    df_datasets = df_datasets.set_index("name")

    # finally launch the code
    convex_hull_stats.openml_stats_all(df_datasets)

    # snapshot = tracemalloc.take_snapshot()
    #
    # sys.stdout = open("profile.txt", "w")
    # display_top(snapshot, limit=100)


if __name__ == "__main__":
    sys.exit(main())
