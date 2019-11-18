import lazygrid as lg
import convex_hull_stats
import sys


def main():
    datasets = lg.fetch_datasets(task="classification", min_classes=2)
    results = convex_hull_stats.openml_stats_all(datasets)
    results.to_csv("./results.csv")


if __name__ == "__main__":
    sys.exit(main())
