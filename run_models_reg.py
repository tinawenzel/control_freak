import sklearn
import pandas as pd
import sys
from sklearn.metrics import mean_absolute_error
from time import time

# h2o requires target variable to be part of dataframe - other libraries don't
X_train = pd.read_csv('data/X_train_reg.csv')
y_train = X_train.target
feature_cols = X_train.drop(['target'],axis=1).columns.tolist()
X_valid = pd.read_csv('data/X_valid_reg.csv')
y_valid = X_valid.target

type_id = sys.argv[1:][0]

print(type_id)

if type_id=='sklearn':
	import sklearn
	v = sys.argv[1:][1]
	from sklearn.ensemble import GradientBoostingRegressor
	model = GradientBoostingRegressor()
	start = time()
	model.fit(X_train[feature_cols],y_train)
	y_pred = model.predict(X_valid[feature_cols])
	end = time()
	score = mean_absolute_error(y_valid, y_pred)
	print("lib:'sklearn' version: {0} mae: {1} timeit: {2}".format(v,score, (end-start)))

if type_id=='xgboost':
	import xgboost
	v = sys.argv[1:][1]
	from xgboost import XGBRegressor
	model = XGBRegressor()
	start = time()
	model.fit(X_train[feature_cols], y_train)
	y_pred = model.predict(X_valid[feature_cols])
	end = time()
	score = mean_absolute_error(y_valid, y_pred)
	print("lib:xgboost version: {0} mae: {1} timeit: {2}".format(v,score, (end-start)))

if type_id=='h2o':
	import h2o
	v = sys.argv[1:][1]
	from h2o.estimators.gbm import H2OGradientBoostingEstimator
	h2o.init()
	h2o.remove_all()

	x = X_train.columns.difference([X_train.columns[-1]]).tolist()
	y = X_train.columns[-1]

	X_train = h2o.H2OFrame(X_train)
	X_valid = h2o.H2OFrame(X_valid)

	gbm = H2OGradientBoostingEstimator(
	    model_id="gbm_v1",
	    seed=2000000,
	    distribution="gaussian"
	)
	start = time()
	gbm.train(x, y, training_frame=X_train, validation_frame=X_valid)
	y_pred = gbm.predict(X_valid[:-1]).as_data_frame()['predict']
	end = time()
	score = mean_absolute_error(y_valid, y_pred)
	# score = gbm.mae(valid = True)
	print("lib: h2o version: {0} score: {1} timeit: {2}".format(v,score, (end-start)))
	h2o.shutdown(prompt=False)
	
if type_id=='lightgbm':
	import lightgbm
	v = sys.argv[1:][1]
	from lightgbm import LGBMRegressor
	model = LGBMRegressor()
	start = time()
	model.fit(X_train[feature_cols], y_train)
	y_pred = model.predict(X_valid[feature_cols])
	end = time()
	score = mean_absolute_error(y_valid, y_pred)
	print("lib: lightgbm version: {0} mae: {1} timeit: {2}".format(v,score, (end-start)))

if type_id=='catboost':
	import catboost
	v = sys.argv[1:][1]
	from catboost import CatBoostRegressor
	model = CatBoostRegressor()
	start = time()
	model.fit(X_train[feature_cols], X_train)
	end = time()
	y_pred = model.predict(X_valid[feature_cols])
	score = mean_absolute_error(y_valid, y_pred)
	print("lib: catboost version: {0} mae: {1} timeit: {2}".format(v,score, (end-start)))

with open("results_reg.txt", "a") as myfile:
    myfile.write(','.join((str(type_id),str(v),str(score),str(end-start))) + "\n")


