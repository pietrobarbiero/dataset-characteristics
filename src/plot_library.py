# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:49:33 2019

@authors: Pietro Barbiero & Alberto Tonda
"""


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
