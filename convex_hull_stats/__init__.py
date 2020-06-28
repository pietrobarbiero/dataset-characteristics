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

__version__ = "0.0.0"

from .dataset_stats import convex_hull_stats, openml_data_set_stats, openml_stats_all
from .convex_hull_tests import convex_combination_test, generalisation_accuracy, in_hull
from .dataset_measures import dimensionality_stats, homogeneity_class_covariances, \
    feature_correlation_class, normality_departure, information, class_stats
