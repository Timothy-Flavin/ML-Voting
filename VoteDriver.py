# Grass fed libraries
import ML_Democracy as MLD
# Factory farmed libraries ;)
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf         
from tensorflow import keras
# Spicy models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as sk_logistic
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier
# Spicier preprocessing techniques 
from sklearn.decomposition import PCA
# Super Secret Constants for test purposes ;) 
TRAIN_LEN = 60000
TEST_LEN  = 10000

# Data to test with
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(f'original shape upon import, x: {x_train.shape}, y: {y_train.shape}')

x_train = np.reshape(x_train, (60000, 28*28))[:TRAIN_LEN]
x_test = np.reshape(x_test, (10000, 28*28))[:TEST_LEN]
y_train = y_train[:TRAIN_LEN]
y_test = y_test[:TEST_LEN]

print(f'new shape after flattening, x: {x_train.shape}, y: {y_train.shape}')

# Preprocessing
def principal_squish_squash(data):
  tran = PCA(n_components=0.950)
  data_out = tran.fit_transform(data)
  return tran, data_out

# Comment this out to not use PCA
tran, x_train = principal_squish_squash(x_train)
x_test = tran.transform(x_test)
# End of PCA Code

print(f'new shape after preprocessing, x: {x_train.shape}, y: {y_train.shape}')

def mlp_init(x_train, y_train, x_test, y_test):
  mlp = MLPClassifier(hidden_layer_sizes=(256,128,32,), alpha=0.01, max_iter=60).fit(x_train, y_train)
  test_score = mlp.score(x_test, y_test)
  train_score = mlp.score(x_train, y_train)

  return train_score, test_score, mlp

def lda_init(x_train, y_train, x_test, y_test):
  lda = LinearDiscriminantAnalysis().fit(x_train, y_train)
  test_score = lda.score(x_test, y_test)
  train_score = lda.score(x_train, y_train)

  return train_score, test_score, lda 

def nb_init(x_train, y_train, x_test, y_test):
  nb = GaussianNB().fit(x_train, y_train)
  test_score = nb.score(x_test, y_test)
  train_score = nb.score(x_train, y_train)

  return train_score, test_score, nb 

def log_init(x_train, y_train, x_test, y_test):
  log = sk_logistic(max_iter=300).fit(x_train, y_train)
  test_score = log.score(x_test, y_test)
  train_score = log.score(x_train, y_train)

  return train_score, test_score, log 
  
def svm_init(x_train, y_train, x_test, y_test):
  svmp = svm.SVC(tol=0.075).fit(x_train, y_train)
  test_score = svmp.score(x_test, y_test)
  train_score = svmp.score(x_train, y_train)

  return train_score, test_score, svmp 

def rf_init(x_train, y_train, x_test, y_test):
  rf = RandomForestClassifier(n_estimators=250).fit(x_train, y_train)
  test_score = rf.score(x_test, y_test)
  train_score = rf.score(x_train, y_train)

  return train_score, test_score, rf 

def gbc_init(x_train, y_train, x_test, y_test):
  gbc = GradientBoostingClassifier(
    max_depth=4,
    n_estimators=15,
    learning_rate=0.125
  ).fit(x_train, y_train)
  test_score = gbc.score(x_test, y_test)
  train_score = gbc.score(x_train, y_train)

  return train_score, test_score, gbc 

def xgb_init(x_train, y_train, x_test, y_test):
  xgbc = XGBClassifier().fit(x_train, y_train)
  test_score = xgbc.score(x_test, y_test)
  train_score = xgbc.score(x_train, y_train)
  return train_score, test_score, xgbc 
# Will be different for tf models and any other library
# They will need to transform the data and possibly not 
# do one hot encoding. 
def predict(model, x):
  temp = model.predict(x)
  return temp

MLD.set_num_classifications(10)
MLD.set_default_data(x_train, x_test, y_train, y_test)
MLD.add_algo(MLD.ML_Algo(log_init, predict, "logistic reg")) # can add a cutom dataset
MLD.add_algo(MLD.ML_Algo(mlp_init, predict, "mlp classifier"))
MLD.add_algo(MLD.ML_Algo(lda_init, predict, "lda"))
MLD.add_algo(MLD.ML_Algo(nb_init, predict,  "Nbayes"))
MLD.add_algo(MLD.ML_Algo(svm_init, predict, "support vector"))
MLD.add_algo(MLD.ML_Algo(rf_init, predict, "random forest"))
MLD.add_algo(MLD.ML_Algo(gbc_init, predict, "gradient boost"))
MLD.add_algo(MLD.ML_Algo(xgb_init, predict, "Ex gradient boost"))
MLD.train_algos(bag=True, featureProportion=0.33) # add nullable args for funnel or not
MLD.current_algos()
MLD.validate_voting(x_test, y_test, method=0) # change this back to validate, when training set function to vote or not
MLD.validate_voting(x_test, y_test, method=1) # change this back to validate, when training set function to vote or not
MLD.validate_voting(x_test, y_test, method=2) # change this back to validate, when training set function to vote or not
MLD.validate_funnel(x_train, y_train, x_test, y_test, MLPClassifier(hidden_layer_sizes=(512,256,64,32,), alpha=0.01, max_iter=50), name="Mlp stack")
MLD.validate_funnel(x_train, y_train, x_test, y_test, svm.SVC(tol=0.075), name="Svc stack")
