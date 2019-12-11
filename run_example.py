import lazygrid as lg
import convex_hull_stats
import sys


def main():
    datasets = lg.datasets.fetch_datasets(task="classification", min_classes=2)
    convex_hull_stats.openml_stats_all(datasets)


if __name__ == "__main__":
    sys.exit(main())
