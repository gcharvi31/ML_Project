#!/usr/bin/python

from __future__ import print_function

import numpy as np
import pandas as pd

from time import time
from tqdm import tqdm

from scipy.stats import boxcox
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, \
	roc_auc_score
from sklearn.preprocessing import RobustScaler, \
	StandardScaler, MinMaxScaler

""" Helper functions for preprocessing, data
	manipulation and model evaluation. """

# Preprocessing

def binarize_columns(df):
	""" If there are only 2 unique values
		in the df then change them to 0 and 1. """
	
	print('Binarizing columns.')
	
	for col in df.columns:
		if df[col].nunique() == 2:
			a, b = df[col].unique()
			df[col].replace({a: 0, b: 1}, inplace=True)

	return df

def drop_zero_var_columns(df):
	""" Drop columns where all values are the same. """
	
	print('Dropping columns with zero variance.')
	
	# Drop all columns with same value throughout
	cols = list(df)
	nunique = df.apply(pd.Series.nunique)
	cols_to_drop = nunique[nunique == 1].index
	df.drop(cols_to_drop, axis=1, inplace=True)
	
def drop_sparse_binary(df, sparsity_limit=30):
	""" Take columns with only 2 unique
		values and if there are less than
		sparsity_limit number of values of the 
		second class, then drop those columns. """
	
	print('Dropping sparse binary columns.')
	
	for col in df.columns:
		if df[col].nunique() == 2:
			a, b = df[col].value_counts()
			if b <= sparsity_limit:
				df.drop([col], axis=1, inplace=True)
				
def boxcox_transform(df, eta=1., skew_threshold=1.):
	""" Apply Boxcox transformation to df to
		remove skewed columns. A copy of the 
		original df is returned. """
	
	df_pos = df
	for i in tqdm(range(len(df_pos.columns)),
				desc='Applying Boxcox transformation',
				unit='column'):
		col = df_pos.columns[i]
		if np.abs(df_pos[col].skew()) > skew_threshold:
			if not (df_pos[col] > 0).all():
				m = min(df_pos[col])
				df_pos[col] += -1*m + eta
			df_pos[col], _lambda = boxcox(df_pos[col])
			
	return df_pos
			
def normalize(df, scaler='robust'):
	""" Normalize dataframe. Scaler can be 
		'robust', 'standard' or 'min_max'. """
	
	print('Normalizing df using ' + scaler + ' scaler.')
	
	# Types of scalers
	scalers = {'robust': RobustScaler(),
				'standard': StandardScaler(),
				'min_max': MinMaxScaler()}
	
	# Transform
	transformer = scalers[scaler]
	X_scaled = transformer.fit_transform(df)
	df = pd.DataFrame(X_scaled, columns=df.columns)
	
	return df


# Data Manipulation

def df_to_float32(df):
	""" Convert all columns of dataframe to float32. """
	
	for i in tqdm(range(len(df.columns)),
				desc='Reading as float32',
				unit='column'):
		col = df.columns[i]
		df[col] = df[col].astype('float32')
	
	return df
		
def replace_inf(df, val=1.0):
	""" Replace all inf with val in df. """
	
	print('Replacing inf in df.')
	
	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.fillna(val, inplace=True)
	
	
# Model evaluation
	
def k_fold(clf, X, y, k=5, scoring='auroc'):
	""" Perform k-fold cross validation by
		calculating AUROC. """
	
	print ('Performing ' + str(k) + '-fold cv...')
	
	if scoring == 'auroc':
		scorer = make_scorer(roc_auc_score)
	else:
		raise NotImplementedError('Metric not supported yet.')
		
	auroc = np.mean(cross_val_score(clf, X, y, cv=5, scoring=scorer))
	print ('Mean AUROC =', auroc)