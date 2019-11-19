import lazygrid as lg
import convex_hull_stats
import sys


def main():
    logger = lg.initialize_logging(log_name="dataset-characteristics")
    datasets = lg.fetch_datasets(task="classification", min_classes=2, logger=logger)
    results = convex_hull_stats.openml_stats_all(datasets, logger=logger)
    results.to_csv("./results.csv")


if __name__ == "__main__":
    sys.exit(main())
