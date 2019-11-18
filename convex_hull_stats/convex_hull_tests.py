# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero and Alberto Tonda
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from scipy.optimize import linprog


def convex_combination_test(X_trainval, X_test):

    out_indeces = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        try:
            out_indeces[i] = not in_hull(X_trainval, X_test[i, :])
        except ValueError:
            out_indeces[i] = True
    out_indeces = out_indeces.astype('bool')

    return out_indeces


def generalisation_accuracy(model, out_indeces, X_test, y_test):

    if sum(out_indeces==False) > 0:
        in_pred = model.predict(X_test[out_indeces==False, :])
        in_acc = accuracy_score(y_test[out_indeces==False], in_pred)
    else:
        in_acc = np.nan

    if sum(out_indeces==True) > 0:
        out_pred = model.predict(X_test[out_indeces==True, :])
        out_acc = accuracy_score(y_test[out_indeces==True], out_pred)
    else:
        out_acc = np.nan

    return in_acc, out_acc


def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

#
# Alternative solutions for computing the convex hull
#


def in_hull_svm(points, x):
    X = np.r_[points, np.expand_dims(x, 0)]
    y = np.zeros(X.shape[0])
    y[-1] = 1
    svm = LinearSVC(C=np.inf, class_weight='balanced', random_state=42)
    svm.fit(X, y)
    score = svm.score(X, y)
    return score < 1


def convex_combination_test_svm(model, X_trainval, X_test, y_test):

    test_out = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        test_out[i] = not in_hull_svm(X_trainval, X_test[i, :])
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

    pca = PCA()
    pca.fit(X_trainval)
    X_trainval_pca = pca.transform(X_trainval)
    X_test_pca = pca.transform(X_test)

    # find parallelepiped around training data and compute the volume
    lims = np.zeros((X_trainval_pca.shape[1], 2))
    for j in range(0, X_trainval_pca.shape[1]):
        lims[j, 0] = np.min(X_trainval_pca[:, j])
        lims[j, 1] = np.max(X_trainval_pca[:, j])
    #volume = np.prod( lims[:, 1]-lims[:, 0] )

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
