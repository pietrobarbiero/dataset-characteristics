#basic libraries
import argparse
import copy
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy.random import RandomState
import random
import os
import sys
import time
import logging
from math import atan2

# tensorflow library
import tensorflow as tf

# sklearn library
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# pandas
from pandas import read_csv

import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull

#matplotlib.rcParams.update({'font.size': 15})

import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import linprog

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def convex_combination_test(model, X_trainval, X_test, y_test):
	
	test_out = np.zeros(X_test.shape[0])
	for i in range(X_test.shape[0]):
		test_out[i] = not in_hull(X_trainval, X_test[i, :])
	test_out = test_out.astype('bool')
	
	if sum(test_out==False) > 0: 
		in_pred = model.predict(X_test[test_out==False, :])
		in_acc = accuracy_score(y_test[test_out==False], in_pred)
	else:
		in_acc = np.nan
	
	if sum(test_out==True) > 0:
		out_pred = model.predict(X_test[test_out==True, :])
		out_acc = accuracy_score(y_test[test_out==True], out_pred)
	else:
		out_acc = np.nan
	
	return test_out, in_acc, out_acc

def pca_test(model, X_trainval, X_test, y_test):
	
	pca = PCA(n_components=X_trainval.shape[1])
	pca.fit(X_trainval)
	X_trainval_pca = pca.transform(X_trainval)
	X_test_pca = pca.transform(X_test)
	
	# find 	parallelepiped around training data and compute the volume
	lims = np.zeros((X_trainval_pca.shape[1], 2))
	for j in range(0, X_trainval.shape[1]):
		lims[j, 0] = np.min(X_trainval_pca[:, j])
		lims[j, 1] = np.max(X_trainval_pca[:, j])
	volume = np.prod( lims[:, 1]-lims[:, 0] )
	
	test_out = np.zeros(X_test_pca.shape[0])
	for j in range(0, X_test_pca.shape[1]):
		test_out = test_out + ( X_test_pca[:, j] < lims[j, 0] ).astype('int')
		test_out = test_out + ( X_test_pca[:, j] > lims[j, 1] ).astype('int')
	test_out = test_out > 0
	
	if sum(test_out==False) > 0: 
		in_pred = model.predict(X_test[test_out==False, :])
		in_acc = accuracy_score(y_test[test_out==False], in_pred)
	else:
		in_acc = np.nan
	
	if sum(test_out==True) > 0:
		out_pred = model.predict(X_test[test_out==True, :])
		out_acc = accuracy_score(y_test[test_out==True], out_pred)
	else:
		out_acc = np.nan
	
	return test_out, in_acc, out_acc, lims

def make_meshgrid(x, y, h=.02):
	k = 1
	x_min, x_max = x.min() - k, x.max() + k
	y_min, y_max = y.min() - k, y.max() + k
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						np.arange(y_min, y_max, h))
	return xx, yy

def plot_contours2(clf, xx, yy, X_contour_out, cmap, **params):
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z[X_contour_out] = 3
	Z = Z.reshape(xx.shape)
	out = plt.contourf(xx, yy, Z, cmap=cmap, label='?', **params)
	return out, Z

def plot_contours(clf, xx, yy, cmap, **params):
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = plt.contourf(xx, yy, Z, cmap=cmap, **params)
	return out, Z


def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""
	
	formatter = logging.Formatter('%(asctime)s %(message)s')
	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)
	
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	
	return logger

#def in_hull(p, hull):
#    """
#    Test if points in `p` are in `hull`
#
#    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
#    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
#    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
#    will be computed
#    """
#    from scipy.spatial import Delaunay
#    if not isinstance(hull,Delaunay):
#        hull = Delaunay(hull)
#
#    return hull.find_simplex(p)>=0

from operator import sub
def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio

def loadMNIST():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	
	X = np.concatenate((x_train, x_test))
	X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[1]))
	y = np.concatenate((y_train, y_test))
	
	return X, y

def main(selectedDataset = "digits"):
	
	# a few hard-coded values
	figsize = [5, 4]
	seed = 42
	selectedClassifiers = ["SVC"]
	n_splits = 10

	# a list of classifiers
	allClassifiers = [
#			[RandomForestClassifier, "RandomForestClassifier", 1],
#			[BaggingClassifier, "BaggingClassifier", 1],
			[SVC, "SVC", 1],
#			[RidgeClassifier, "RidgeClassifier", 1],
#			[AdaBoostClassifier, "AdaBoostClassifier", 1],
#			[ExtraTreesClassifier, "ExtraTreesClassifier", 1],
#			[GradientBoostingClassifier, "GradientBoostingClassifier", 1],
#			[SGDClassifier, "SGDClassifier", 1],
#			[PassiveAggressiveClassifier, "PassiveAggressiveClassifier", 1],
#			[LogisticRegression, "LogisticRegression", 1],
			]
	
	selectedClassifiers = [classifier[1] for classifier in allClassifiers]
	
	folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-chull-pca-" + selectedDataset
	if not os.path.exists(folder_name) : 
		os.makedirs(folder_name)
	else :
		sys.stderr.write("Error: folder \"" + folder_name + "\" already exists. Aborting...\n")
		sys.exit(0)
	# open the logging file
	logfilename = os.path.join(folder_name, 'logfile.log')
	logger = setup_logger('logfile_' + folder_name, logfilename)
	logger.info("All results will be saved in folder \"%s\"" % folder_name)

	# load different datasets, prepare them for use
	logger.info("Preparing data...")
	# synthetic databases
	centers = [[1, 1], [-1, -1], [1, -1]]
	blobs_X, blobs_y = make_blobs(n_samples=400, centers=centers, n_features=2, cluster_std=0.6, random_state=seed)
	circles_X, circles_y = make_circles(n_samples=400, noise=0.15, factor=0.4, random_state=seed)
	moons_X, moons_y = make_moons(n_samples=400, noise=0.2, random_state=seed)
	iris = datasets.load_iris()
	digits = datasets.load_digits()
	mnist_X, mnist_y = loadMNIST() # local function

	dataList = [
			[blobs_X, blobs_y, 0, "blobs"],
			[circles_X, circles_y, 0, "circles"],
			[moons_X, moons_y, 0, "moons"],
	        [iris.data, iris.target, 0, "iris4"],
	        [iris.data[:, 2:4], iris.target, 0, "iris2"],
	        [digits.data, digits.target, 0, "digits"],
			[mnist_X, mnist_y, 0, "mnist"]
		      ]

	# argparse; all arguments are optional
	parser = argparse.ArgumentParser()

	parser.add_argument("--classifiers", "-c", nargs='+', help="Classifier(s) to be tested. Default: %s. Accepted values: %s" % (selectedClassifiers[0], [x[1] for x in allClassifiers]))
	parser.add_argument("--dataset", "-d", help="Dataset to be tested. Default: %s. Accepted values: %s" % (selectedDataset,[x[3] for x in dataList]))

	# finally, parse the arguments
	args = parser.parse_args()

	# a few checks on the (optional) inputs
	if args.dataset :
		selectedDataset = args.dataset
		if selectedDataset not in [x[3] for x in dataList] :
			logger.info("Error: dataset \"%s\" is not an accepted value. Accepted values: %s" % (selectedDataset, [x[3] for x in dataList]))
			sys.exit(0)

	if args.classifiers != None and len(args.classifiers) > 0 :
		selectedClassifiers = args.classifiers
		for c in selectedClassifiers :
			if c not in [x[1] for x in allClassifiers] :
				logger.info("Error: classifier \"%s\" is not an accepted value. Accepted values: %s" % (c, [x[1] for x in allClassifiers]))
				sys.exit(0)

	# TODO: check that min_points < max_points and max_generations > 0


	# print out the current settings
	logger.info("Settings of the experiment...")
	logger.info("Fixed random seed: %d" %(seed))
	logger.info("Selected dataset: %s; Selected classifier(s): %s" % (selectedDataset, selectedClassifiers))

	# create the list of classifiers
	classifierList = [ x for x in allClassifiers if x[1] in selectedClassifiers ]

	# pick the dataset
	db_index = -1
	for i in range(0, len(dataList)) :
		if dataList[i][3] == selectedDataset :
			db_index = i

	dbname = dataList[db_index][3]

	X, y = dataList[db_index][0], dataList[db_index][1]
	number_classes = np.unique(y).shape[0]

	logger.info("Creating train/test split...")
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#	listOfSplits = [split for split in skf.split(X, y)]
#	trainval_index, test_index = listOfSplits[0]
	
	split_index = 1
	in_acc_cc_list = []
	out_acc_cc_list = []
	in_acc_pca_list = []
	out_acc_pca_list = []
	in_cc_list = []
	out_cc_list = []
	in_pca_list = []
	out_pca_list = []
	for trainval_index, test_index in skf.split(X, y) :
	
		X_trainval, y_trainval = X[trainval_index], y[trainval_index]
		X_test, y_test = X[test_index], y[test_index]
	#	logger.info("Training set: %d lines (%.2f%%); test set: %d lines (%.2f%%)" % (X_trainval.shape[0], (100.0 * float(X_trainval.shape[0]/X.shape[0])), X_test.shape[0], (100.0 * float(X_test.shape[0]/X.shape[0]))))
		logger.info("\tSplit %d" %(split_index))
		
		# rescale data
		scaler = StandardScaler()
		sc = scaler.fit(X_trainval)
		X = sc.transform(X)
		X_trainval = sc.transform(X_trainval)
		X_test = sc.transform(X_test)
		
		# train classifier
		model = copy.deepcopy( SVC(random_state=seed) )
		model.fit(X_trainval, y_trainval)
		
		test_out_cc, in_acc_cc, out_acc_cc = convex_combination_test(model, X_trainval, X_test, y_test)
		test_out_pca, in_acc_pca, out_acc_pca, lims = pca_test(model, X_trainval, X_test, y_test)
		
		if in_acc_cc is not np.nan: in_acc_cc_list.append(in_acc_cc)
		if out_acc_cc is not np.nan: out_acc_cc_list.append(out_acc_cc)
		if in_acc_pca is not np.nan: in_acc_pca_list.append(in_acc_pca)
		if out_acc_pca is not np.nan: out_acc_pca_list.append(out_acc_pca)
		test_size = len(y_test)
		in_cc_list.append(sum(test_out_cc==False)/test_size)
		out_cc_list.append(sum(test_out_cc==True)/test_size)
		in_pca_list.append(sum(test_out_pca==False)/test_size)
		out_pca_list.append(sum(test_out_pca==True)/test_size)
	
		pred = model.predict(X_test)
		errs = y_test != pred
		
		logger.info("\t\tUsing convex combinations of training samples")
		logger.info('\t\t\tInterpolation accuracy (#samples: %d) = %.4f' %(sum(test_out_cc==False), in_acc_cc))
		logger.info('\t\t\tExtrapolation accuracy (#samples: %d) = %.4f' %(sum(test_out_cc==True), out_acc_cc))
				
		logger.info("\t\tUsing PCA upper bound approximation")
		logger.info('\t\t\tInterpolation accuracy (#samples: %d) = %.4f' %(sum(test_out_pca==False), in_acc_pca))
		logger.info('\t\t\tExtrapolation accuracy (#samples: %d) = %.4f' %(sum(test_out_pca==True), out_acc_pca))
		
		if X.shape[1] == 2:
		
			# apply PCA
			pca = PCA(n_components=X.shape[1])
			pca.fit(X_trainval)
			pca2 = PCA(n_components=2)
			pca2.fit(X_trainval)
			
			X_trainval_pca2 = pca2.transform(X_trainval)
			X_trainval_pca = pca.transform(X_trainval)
			X_pca = pca.transform(X)
			X_test_pca = pca.transform(X_test)
			
			if True:
				# now, the way the decision boundary stuff works is by creating
				# a grid over all the space of the features, and asking for predictions
				# all around the place
				Nx = Ny = 100 # 300 x 300 grid
				k = 0.4
				x_max = X_pca[:,0].max() + k
				x_min = X_pca[:,0].min() - k
				y_max = X_pca[:,1].max() + k
				y_min = X_pca[:,1].min() - k
			
				xgrid = np.arange(x_min, x_max, 1. * (x_max-x_min) / Nx)
				ygrid = np.arange(y_min, y_max, 1. * (y_max-y_min) / Ny)
			
				xx, yy = np.meshgrid(xgrid, ygrid)
				X_full_grid = np.array(list(zip(np.ravel(xx), np.ravel(yy))))
				
				out1 = np.zeros(X_full_grid.shape[0])
				for i in range(X_full_grid.shape[0]):
					out1[i] = not in_hull(X_trainval_pca2, X_full_grid[i, :])
				out1 = out1.astype('bool')
				
				X_full_grid_inverse = pca2.inverse_transform(X_full_grid)
				Yp = model.predict(X_full_grid_inverse)
				Yp[out1] = -1
				
				# get decision boundary line
				Yp = Yp.reshape(xx.shape)
				Yb1 = np.zeros(xx.shape)
				
				Yb1[:-1, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb1[:-1, :])
				Yb1[1:, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb1[1:, :])
				Yb1[:, :-1] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb1[:, :-1])
				Yb1[:, 1:] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb1[:, 1:])
				
				out2 = np.zeros(X_full_grid.shape[0])
				for j in range(0, X_full_grid.shape[1]):
					out2 = out2 + ( X_full_grid[:, j] < lims[j, 0] ).astype('int')
					out2 = out2 + ( X_full_grid[:, j] > lims[j, 1] ).astype('int')
				out2 = out2 > 0
				
				X_full_grid_inverse = pca2.inverse_transform(X_full_grid)
				Yp = model.predict(X_full_grid_inverse)
				Yp[out2] = -1
				
				# get decision boundary line
				Yp = Yp.reshape(xx.shape)
				Yb2 = np.zeros(xx.shape)
				
				Yb2[:-1, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb2[:-1, :])
				Yb2[1:, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb2[1:, :])
				Yb2[:, :-1] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb2[:, :-1])
				Yb2[:, 1:] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb2[:, 1:])
			
			cmap = ListedColormap(sns.color_palette("bright", number_classes).as_hex())
			
			fig = plt.figure(figsize=figsize)
			ax = fig.add_subplot(111)
			ax.scatter(X_trainval_pca[:, 0], X_trainval_pca[:, 1], c=y_trainval, cmap=cmap, marker='s', alpha=0.1, label="train")
			ax.scatter(X_test_pca[test_out_cc==True, 0], X_test_pca[test_out_cc==True, 1], facecolors='none', edgecolors='r', marker='D', alpha=0.8, label="out")
			ax.scatter(X_test_pca[errs, 0], X_test_pca[errs, 1], c='k', marker='x', alpha=1, label="errors")
			ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.5, label="test")
			aspect_ratio = get_aspect(ax)
			ax.imshow(Yb1, origin='lower', interpolation=None, cmap='Greys', extent=[x_min, x_max, y_min, y_max], alpha=1.0)
			ax.set_aspect(aspect_ratio)
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4)
			ax.set_title('CC: in-acc (%d) = %.4f - out-acc (%d) = %.4f' \
				   %(sum(test_out_cc==False), in_acc_cc, sum(test_out_cc==True), out_acc_cc))
			plt.tight_layout()
			plt.savefig(os.path.join(folder_name, '%s_closed_db_cc_split_%2d.png' %(dbname, split_index)))
			plt.draw()
			
			
			fig = plt.figure(figsize=figsize)
			ax = fig.add_subplot(111)
			ax.scatter(X_trainval_pca[:, 0], X_trainval_pca[:, 1], c=y_trainval, cmap=cmap, marker='s', alpha=0.1, label="train")
			ax.scatter(X_test_pca[test_out_pca==True, 0], X_test_pca[test_out_pca==True, 1], facecolors='none', edgecolors='r', marker='s', alpha=0.8, label="out")
			ax.scatter(X_test_pca[errs, 0], X_test_pca[errs, 1], c='k', marker='x', alpha=1, label="errors")
			ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.5, label="test")
			aspect_ratio = get_aspect(ax)
			ax.imshow(Yb2, origin='lower', interpolation=None, cmap='Greys', extent=[x_min, x_max, y_min, y_max], alpha=1.0)
			ax.set_aspect(aspect_ratio)
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4)
			ax.set_title('PCA: in-acc (%d) = %.4f - out-acc (%d) = %.4f' \
						%(sum(test_out_pca==False), in_acc_pca, sum(test_out_pca==True), out_acc_pca))
			plt.tight_layout()
			plt.savefig(os.path.join(folder_name, '%s_closed_db_pcasplit_%02d.png' %(dbname, split_index)))
			plt.draw()
		
		split_index = split_index + 1
	
	logger.info("Using convex combinations of training samples")
	logger.info("\tFraction of test samples inside the convex polytope: %.4f (+- %.4f)" %( np.mean(in_cc_list), np.std(in_cc_list) / np.sqrt(n_splits) ))
	logger.info("\tFraction of test samples outside the convex polytope: %.4f (+- %.4f)" %( np.mean(out_cc_list), np.std(out_cc_list) / np.sqrt(n_splits) ))
	logger.info('\tInterpolation accuracy: %.4f (+- %.4f)'  %( np.mean(in_acc_cc_list), np.std(in_acc_cc_list) / np.sqrt(n_splits) ))
	logger.info('\tExtrapolation accuracy: %.4f (+- %.4f)'  %( np.mean(out_acc_cc_list), np.std(out_acc_cc_list) / np.sqrt(n_splits) ))
			
	logger.info("Using PCA upper bound approximation")
	logger.info("\tFraction of test samples inside the convex polytope: %.4f (+- %.4f)" %( np.mean(in_pca_list), np.std(in_pca_list) / np.sqrt(n_splits) ))
	logger.info("\tFraction of test samples outside the convex polytope: %.4f (+- %.4f)" %( np.mean(out_pca_list), np.std(out_pca_list) / np.sqrt(n_splits) ))
	logger.info('\tInterpolation accuracy: %.4f (+- %.4f)'  %( np.mean(in_acc_pca_list), np.std(in_acc_pca_list) / np.sqrt(n_splits) ))
	logger.info('\tExtrapolation accuracy: %.4f (+- %.4f)'  %( np.mean(out_acc_pca_list), np.std(out_acc_pca_list) / np.sqrt(n_splits) ))
	
	logger.handlers.pop()
	return


if __name__ == "__main__" :
	
	dataList = [
#		["iris4"],
#		["iris2"],
#		["moons"],
#		["blobs"],
#		["circles"],
		["digits"],
		["mnist"],
		]
	for dataset in dataList:
		main(dataset[0])
	sys.exit()
	
