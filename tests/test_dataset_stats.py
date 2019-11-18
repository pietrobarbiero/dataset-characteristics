import unittest


class TestDatasetStats(unittest.TestCase):

    def test_main(self):

        import lazygrid as lg
        import convex_hull_stats

        datasets = lg.fetch_datasets(task="classification", min_classes=2, max_samples=1000, max_features=10)

        results = convex_hull_stats.openml_stats_all(datasets)

        results.to_csv("./results.csv")

        self.assertEqual(len(results), 480)


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasetStats)
unittest.TextTestRunner(verbosity=2).run(suite)
