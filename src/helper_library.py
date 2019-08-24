# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:49:53 2019

@authors: Pietro Barbiero & Alberto Tonda
"""

import numpy as np
import logging
import datetime
import os
import sys
import openml
import csv

from difflib import SequenceMatcher

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

allClassifiers = [
		#[RandomForestClassifier, "RandomForest5", 1],
		[RandomForestClassifier, "RandomForest10", 10],
		#[BaggingClassifier, "Bagging", 1],
		[SVC, "SVC", 1],
		#[RidgeClassifier, "Ridge", 1],
		[LogisticRegression, "LogisticRegression", 1],
]

def setup():
	
	# create experiment folder
	folder_name = setup_folder()
	
	# open the logging file
	logfilename = os.path.join(folder_name, 'logfile.log')
	logger = setup_logger('logfile_' + folder_name, logfilename)
	logger.info("All results will be saved in folder \"%s\"" % folder_name)
	
	csv_file = setup_csv(folder_name)
	
	return logger, folder_name, csv_file

def setup_csv(folder_name):
	
	csv_file = os.path.join(folder_name, "convex_hull_results.csv")
	csv_columns = ['db_name', 'n_samples', 'n_features', 'n_classes', \
					'intrinsic_dim', 'intrinsic_dim_ratio', 'feature_noise', \
					'distance_mean', 'distance_std', \
					'levene_stat', 'levene_pval', 'levene_success', \
					'feature_correlation_mean', \
					'skew_mean', 'kurtosis_mean', \
					'mi_mean', \
					'in_test', 'ex_test', 'n_splits', 'shuffle', 'seed', 'out_indeces']
	
	accuracy_cols = ['train_acc', 'in_acc', 'out_acc']
	
	for classifier in allClassifiers:
		_, classifier_name, _ = classifier
		csv_columns.extend([classifier_name + '_' + s for s in accuracy_cols])
	
	with open(csv_file, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
		writer.writeheader()
	
	return csv_file

def setup_folder():
	folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-convexhull"
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	else:
		sys.stderr.write("Error: folder \"" + folder_name + "\" already exists. Aborting...\n")
		sys.exit(0)
	return folder_name

def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""
	
	formatter = logging.Formatter('%(asctime)s %(message)s')
	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)
	
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	
	return logger

def isinteger(x):
	return np.sum( np.equal(np.mod(x, 1), 0) ) == len(x)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_classifiers():
	return allClassifiers

def get_data(logger, dataset_name, did):
	
	err = None
	db = openml.datasets.get_dataset(did)
	
	try:
		X, y, attribute_names = db.get_data(
				   target=db.default_target_attribute,
				      return_attribute_names=True)
	except openml.exceptions.PyOpenMLError as err:
		logger.exception("PyOpenML cannot handle string features")
		return 3*[err]
			
	if not isinstance(X, np.ndarray):
		X = X.toarray()
		
	si = SimpleImputer(missing_values=np.nan, strategy='mean')
	X = si.fit_transform(X)
	
	le = LabelEncoder()
	y = le.fit_transform(y)
	n_classes = np.unique(y).shape[0]
	
	for feature in X.T:
		if isinteger(feature):
			err = ValueError
			return 3*[err]
	
	return X, y, n_classes

def get_datasets(logger):
	
	db_list = []
	db_out = []
	data = []
	
	for key, db in openml.datasets.list_datasets().items():
		
		try:
			flag = 0
			if len(db_list) > 0:
				for db_name in db_list:
					if len(db['name']) < 4:
						r = 0.6
					elif len(db['name']) < 10:
						r = 0.7
					else:
						r = 0.8
					if similar(db_name, db['name']) > r:
						flag = 1
						db_out.append([ db_name, db['name'] ])
						break
			
			if ( db['NumberOfClasses'] > 0 and db['NumberOfSymbolicFeatures'] == 1 ) and \
					db['NumberOfInstances'] < 10000 and flag == 0 and db['NumberOfFeatures'] < 10:
			
				data.append([ db['name'], db['did'], db['NumberOfInstances'], db['NumberOfFeatures'] ])
				db_list.append(db['name'])
				
		except KeyError:
			logger.exception("Key not found in database number " + str(db['did']))
			
	data.sort(key=lambda x: x[3])
	
	return data
	