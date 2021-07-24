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

import unittest

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TestDatasetStats(unittest.TestCase):

    def test_main(self):

        import convex_hull_stats

        X, y = load_iris(return_X_y=True)
        classifiers = [Pipeline([("LogisticRegression", LogisticRegression(random_state=42))])]
        convex_hull_stats.compute_dataset_stats(X, y, classifiers)

        print()


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasetStats)
unittest.TextTestRunner(verbosity=2).run(suite)
