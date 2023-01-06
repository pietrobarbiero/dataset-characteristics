# Generalization in Machine Learning as a Function of Dataset Characteristics #

Scripts to generate all the data and figures used in the paper "Generalization in Machine Learning as a Function of Dataset Characteristics" [[arXiv:2006.15680](https://arxiv.org/abs/2006.15680)]

## TODO
Analysis on mnist_784 crashes mysteriously with a segmentation fault. Here is the description of the segmentation fault obtained through the faulthandler module:

Current thread 0x00007fd29a297740 (most recent call first):
  File "/home/squillero/anaconda3/lib/python3.7/site-packages/scipy/linalg/decomp_svd.py", line 126 in svd
  File "/home/squillero/anaconda3/lib/python3.7/site-packages/scipy/optimize/_remove_redundancy.py", line 400 in _remove_redundancy_svd
  File "/home/squillero/anaconda3/lib/python3.7/site-packages/scipy/optimize/_linprog_util.py", line 852 in _presolve
  File "/home/squillero/anaconda3/lib/python3.7/site-packages/scipy/optimize/_linprog.py", line 625 in linprog
  File "/home/squillero/dataset-characteristics/convex_hull_stats/convex_hull_tests.py", line 68 in in_hull
  File "/home/squillero/dataset-characteristics/convex_hull_stats/convex_hull_tests.py", line 35 in convex_combination_test
  File "/home/squillero/dataset-characteristics/convex_hull_stats/dataset_stats.py", line 65 in cross_validation
  File "/home/squillero/dataset-characteristics/convex_hull_stats/dataset_stats.py", line 137 in compute_dataset_stats
  File "/home/squillero/dataset-characteristics/convex_hull_stats/dataset_stats.py", line 196 in openml_stats_all
  File "run_example.py", line 126 in main
  File "run_example.py", line 135 in <module>

So, the issue is in scipy.linalg.decomp_svd on line 126, in correspondance of a call to gesXd, a function obtained calling:
gesXd, gesXd_lwork = get_lapack_funcs(funcs, (a1,), ilp64='preferred')
