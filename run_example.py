import lazygrid as lg
import convex_hull_stats
import sys
import linecache
import os
import tracemalloc
import logging


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

    datasets = lg.datasets.fetch_datasets(task="classification", min_classes=2, max_features=4000, update_data=True)
    convex_hull_stats.openml_stats_all(datasets)

    # snapshot = tracemalloc.take_snapshot()
    #
    # sys.stdout = open("profile.txt", "w")
    # display_top(snapshot, limit=100)


if __name__ == "__main__":
    sys.exit(main())
