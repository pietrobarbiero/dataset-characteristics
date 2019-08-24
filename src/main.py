# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:42:31 2019

@authors: Pietro Barbiero & Alberto Tonda
"""

# convex hull libraries
from convex_hull_library import convex_combination_test, generalisation_accuracy
from database_measures_library import homogeneity_class_covariances, \
	feature_correlation_class, normality_departure, information, \
	dimensionality_stats
from helper_library import setup, get_classifiers, \
							get_data, get_datasets

#basic libraries
import copy
import numpy as np
import csv
import sys

# sklearn library
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

#import seaborn as sns
#from matplotlib.colors import ListedColormap
#matplotlib.rcParams.update({'font.size': 15})

import warnings
warnings.filterwarnings("ignore")




def main(logger, dataset_name, did, classifiers, seed=42, n_splits=10, shuffle=True):

	# load different datasets, prepare them for use
	print("Preparing data...")
	
	X, y, n_classes = get_data(logger, dataset_name, did)
	if not isinstance(X, np.ndarray):
		return X
	
	# TODO: add argument parser to run main from command line independently
	
	# print out the current settings
	logger.info("Settings of the experiment...")
	logger.info("Fixed random seed: %d" %(seed))
	logger.info("Selected dataset: %s" % (dataset_name))

	logger.info("Creating train/test split...")
	skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
	
	# compute few stats about cross-validation
	train_perc = (n_splits-1)/n_splits
	test_perc = 1/n_splits
	logger.info("Training set: %d lines (%.2f%%); test set: %d lines (%.2f%%); #features: %d" % (
			round(train_perc * X.shape[0]), (n_splits-1)/n_splits*100, 
			round(test_perc * X.shape[0]), 1/n_splits*100,
			X.shape[1] ))
	
	split_index = 1
	results = []
	
	for trainval_index, test_index in skf.split(X, y) :
		
		logger.info("\tSplit %d" %(split_index))
		X_trainval, y_trainval = X[trainval_index], y[trainval_index]
		X_test, y_test = X[test_index], y[test_index]
		
		# rescale data
		scaler = StandardScaler()
		sc = scaler.fit(X_trainval)
		X = sc.transform(X)
		X_trainval = sc.transform(X_trainval)
		X_test = sc.transform(X_test)
		
		# compute training set stats
		levene_stat, levene_pval, levene_success = homogeneity_class_covariances(X_trainval, y_trainval)
		feature_correlation_mean = feature_correlation_class(X_trainval, y_trainval)
		skew_mean, kurtosis_mean = normality_departure(X_trainval, y_trainval)
		mi_mean = information(X_trainval, y_trainval)
		dimensionality, intrinsic_dim, intrinsic_dim_ratio, feature_noise, dist = dimensionality_stats(X_trainval)
		
		# convex hull test
		out_indeces = convex_combination_test(X_trainval, X_test)
		n_points_out = len(y_test) - len(out_indeces)
		n_points_in = len(out_indeces)
		
		results.append([
				# basic properties
				dataset_name,
				X.shape[0],
				dimensionality,
				n_classes,
				
				# dimensionality
				intrinsic_dim,
				intrinsic_dim_ratio,
				feature_noise,
				
				# distances
				np.mean( dist ), # average distance between samples
				np.std( dist ), # average distance between samples
				
				# stats measures
				levene_stat, levene_pval, levene_success,
				feature_correlation_mean,
				skew_mean, kurtosis_mean,
				mi_mean,
				
				# convex-hull
				n_points_in / len(y_test),
				n_points_out / len(y_test),
				n_splits, shuffle, seed,
				out_indeces,
		])
	
	
		for classifier in classifiers:
			
			classifier_type, classifier_name, _ = classifier
			logger.info("\tClassifier %s" %(classifier_name))
			
			# train classifiers
			model = copy.deepcopy( classifier_type(random_state=seed) )
			model.fit(X_trainval, y_trainval)
			train_pred = model.predict(X_trainval)
			
			train_acc = accuracy_score(y_trainval, train_pred)
			
			# convex-hull test
			in_acc_cc, out_acc_cc = generalisation_accuracy(model, out_indeces, X_test, y_test)
			logger.info("\t\tUsing Linear Programming")
			logger.info('\t\t\tInterpolation accuracy (#samples: %d) = %.4f' %(n_points_in, in_acc_cc))
			logger.info('\t\t\tExtrapolation accuracy (#samples: %d) = %.4f' %(n_points_out, out_acc_cc))
			
			# save stats
			results[-1].extend([
				# accuracy
				train_acc,
				in_acc_cc,
				out_acc_cc,
			])
			
		split_index = split_index + 1
	
	return results


if __name__ == "__main__" :

	logger, folder_name, csv_file = setup()
	data = get_datasets(logger)
	classifiers = get_classifiers()
	
	for dataset in data:
		dataset_name, did = dataset[0], dataset[1]
		results = main(logger, dataset_name, did, classifiers, seed=42, n_splits=10)
		
		if isinstance(results, list):
			
			with open(csv_file, 'a', newline='') as csvfile:
				writer = csv.writer(csvfile)
				for r in results:
					writer.writerow(r)
	
	sys.exit(0)
		