# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:47:00 2019

@authors: Pietro Barbiero & Alberto Tonda
"""


import numpy as np

from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

from scipy.stats import levene
from scipy.stats import kurtosis, skew

def dimensionality_stats(X_trainval):
	dimensionality = X_trainval.shape[1]
	pca = PCA()
	pca.fit(X_trainval)
	exp_var = 0
	intrinsic_dim = 0
	for pc in pca.explained_variance_ratio_:
		if exp_var > 0.9:
			break
		exp_var = exp_var + pc
		intrinsic_dim = intrinsic_dim + 1
	
	intrinsic_dim_ratio = intrinsic_dim / dimensionality
	
	feature_noise = ( dimensionality - intrinsic_dim ) / dimensionality
	
	dist = pdist(X_trainval)
	
	return dimensionality, intrinsic_dim, intrinsic_dim_ratio, feature_noise, dist

def homogeneity_class_covariances(X_train, y_train):
	
	dims = X_train.shape[1]
	y_u = np.unique(y_train)
	covs = []
		
	for y in y_u:
		covs.append( X_train[y_train==y] )
		
	levene_stat, levene_pval = [], []
	levene_success = 0
	for j in range(0, dims):
		L = []
		for M in covs:
			L.append( M[:, j] )
		l_stat, l_pval = levene(*L)
		levene_pval.append( l_pval )
		if l_pval < 0.05:
			levene_success += 1
			levene_stat.append( l_stat )
	
	if levene_success > 0:
		levene_stat_avg = np.average( levene_stat, weights=levene_stat )
	else:
		levene_stat_avg = np.nan
	levene_pval_avg = np.average( levene_pval, weights=levene_pval )
	levene_success_ratio = levene_success / dims
	
	return levene_stat_avg, levene_pval_avg, levene_success_ratio

def feature_correlation_class(X_train, y_train):
	
	y_u = np.unique(y_train)
	rho = []
		
	for y in y_u:
		C = np.abs( np.corrcoef( X_train[y_train==y] ) )
		triu = C[np.triu_indices(C.shape[0], k = 1)]
		rho.append( np.average(triu, weights=triu) )
	
	fcc_mean = np.average( rho, weights=rho )
#	fcc_std = np.std( rho )
	
	return fcc_mean

def normality_departure(X_train, y_train):
	
	y_u = np.unique(y_train)
	skew_list = []
	kurtosis_list = []
		
	for y in y_u:
		skew_list.append( np.abs( skew( X_train[y_train==y] ) ) )
		kurtosis_list.append( kurtosis( X_train[y_train==y], fisher=True) )
	
	skew_mean = np.average( skew_list, weights=skew_list )
#	skew_std = np.std( skew_list )
	kurtosis_mean = np.average( kurtosis_list, weights=kurtosis_list )
#	kurtosis_std = np.std( kurtosis_list )
	
	return skew_mean, kurtosis_mean


def information(X_train, y_train):
	
	mi = mutual_info_classif(X_train, y_train)
	mi_mean = np.average( mi, weights=mi )
#	mi_std = np.std( mi )
	
	return mi_mean