import unittest


class TestDatasetStats(unittest.TestCase):

    def test_main(self):

        import lazygrid as lg
        import convex_hull_stats

        datasets = lg.datasets.fetch_datasets(task="classification", min_classes=1, max_samples=300, max_features=6)

        convex_hull_stats.openml_stats_all(datasets)

        print()


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasetStats)
unittest.TextTestRunner(verbosity=2).run(suite)
